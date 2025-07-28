#!/bin/bash
# Start nginx load balancer with multiple uwsgi workers
# Simplified approach using background processes

set -e

echo "Starting multi-worker deployment with nginx..."
echo "Workers: $NUM_WORKERS, Base port: $BASE_PORT, Nginx port: $NGINX_PORT"

# Validate and fix uwsgi configuration
if [ -z "$UWSGI_PROCESSES" ]; then
    UWSGI_PROCESSES=2
fi

if [ -z "$UWSGI_CHEAPER" ]; then
    UWSGI_CHEAPER=1
elif [ "$UWSGI_CHEAPER" -le 0 ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be at least 1"
    UWSGI_CHEAPER=1
    echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
elif [ "$UWSGI_CHEAPER" -ge "$UWSGI_PROCESSES" ]; then
    echo "WARNING: UWSGI_CHEAPER ($UWSGI_CHEAPER) must be lower than UWSGI_PROCESSES ($UWSGI_PROCESSES)"
    if [ "$UWSGI_PROCESSES" -eq 1 ]; then
        # For single process, disable cheaper mode entirely
        echo "Disabling cheaper mode for single process setup"
        UWSGI_CHEAPER=""
    else
        UWSGI_CHEAPER=$((UWSGI_PROCESSES - 1))
        echo "Setting UWSGI_CHEAPER to $UWSGI_CHEAPER"
    fi
fi

export UWSGI_PROCESSES
if [ -n "$UWSGI_CHEAPER" ]; then
    export UWSGI_CHEAPER
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: $UWSGI_CHEAPER"
else
    echo "UWSGI config - Processes: $UWSGI_PROCESSES, Cheaper: disabled"
fi

# Generate upstream servers configuration for nginx
echo "Generating nginx configuration..."

# Write upstream servers to a temp file
UPSTREAM_FILE="/tmp/upstream_servers.conf"
> $UPSTREAM_FILE  # Clear the file
for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT + i - 1))
    echo "        server 127.0.0.1:${PORT} max_fails=3 fail_timeout=30s;" >> $UPSTREAM_FILE
done

echo "Generated upstream servers for $NUM_WORKERS workers:"
cat $UPSTREAM_FILE

# Create nginx config by replacing placeholders
# First replace the NGINX_PORT, then insert the upstream servers
sed "s|\${NGINX_PORT}|${NGINX_PORT}|g" /etc/nginx/nginx.conf.template > /tmp/nginx_temp.conf

# Replace the upstream servers placeholder with the actual servers
# Use a different approach - split at the placeholder and reassemble
awk -v upstream_file="$UPSTREAM_FILE" '
/\${UPSTREAM_SERVERS}/ {
    while ((getline line < upstream_file) > 0) {
        print line
    }
    close(upstream_file)
    next
}
{ print }
' /tmp/nginx_temp.conf > /etc/nginx/nginx.conf

echo "Generated nginx config with upstream servers:"
echo "✅ Nginx configuration created successfully"

# Test nginx configuration
echo "Testing nginx configuration..."
if ! nginx -t; then
    echo "ERROR: nginx configuration test failed"
    echo "Generated nginx.conf:"
    cat /etc/nginx/nginx.conf
    exit 1
fi

# Create log directory
mkdir -p /var/log/nginx

# Start workers as background processes
echo "Starting $NUM_WORKERS workers in parallel..."
WORKER_PIDS=()

# Function to cleanup on exit
cleanup() {
    echo "Shutting down workers and nginx..."
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    pkill -f nginx || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start all workers simultaneously
for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT + i - 1))

    echo "Starting worker $i on port $PORT..."

    # Create a custom uwsgi.ini for this worker that uses HTTP instead of unix socket
    cat > /tmp/worker${i}_uwsgi.ini << EOF
[uwsgi]
module = main
callable = app
processes = ${UWSGI_PROCESSES}
http = 0.0.0.0:${PORT}
master = true
die-on-term = true
memory-report = true
EOF

    # Add cheaper configuration if enabled
    if [ -n "$UWSGI_CHEAPER" ]; then
        echo "cheaper = ${UWSGI_CHEAPER}" >> /tmp/worker${i}_uwsgi.ini
    fi

    echo "Created custom uwsgi config for worker $i (HTTP port $PORT)"

    # Start worker with custom config - NO DELAY between workers
    cd /app && env LISTEN_PORT=$PORT WORKER_NUM=$i uwsgi --ini /tmp/worker${i}_uwsgi.ini > /var/log/worker${i}.log 2>&1 &

    WORKER_PID=$!
    WORKER_PIDS+=($WORKER_PID)

    echo "Worker $i started with PID $WORKER_PID on port $PORT"
done

echo "All $NUM_WORKERS workers started simultaneously - waiting for readiness..."

# Wait for workers to be ready
echo "Waiting for workers to start..."
READY_WORKERS=0
TIMEOUT=180  # Increased timeout since uwsgi takes time to start
START_TIME=$(date +%s)

# Track which workers are ready to avoid redundant checks
declare -A WORKER_READY

while [ $READY_WORKERS -lt $NUM_WORKERS ]; do
    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "ERROR: Timeout waiting for workers to start"

        # Show worker status and logs
        echo "Worker status:"
        for i in "${!WORKER_PIDS[@]}"; do
            pid=${WORKER_PIDS[$i]}
            port=$((BASE_PORT + i))
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker $((i+1)) (PID $pid): Process Running"

                # Check if port is bound (use netstat instead of ss)
                if netstat -tlnp 2>/dev/null | grep ":$port " > /dev/null; then
                    echo "    Port $port: Bound"
                else
                    echo "    Port $port: Not bound"
                fi

                # Show recent log output
                echo "    Recent log output:"
                tail -20 /var/log/worker$((i+1)).log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            else
                echo "  Worker $((i+1)) (PID $pid): Dead"
                echo "    Log:"
                tail -30 /var/log/worker$((i+1)).log 2>/dev/null | sed 's/^/      /' || echo "      No log found"
            fi
        done

        exit 1
    fi

    READY_WORKERS=0
    for i in $(seq 1 $NUM_WORKERS); do
        # Skip workers that are already ready
        if [ "${WORKER_READY[$i]}" = "1" ]; then
            READY_WORKERS=$((READY_WORKERS + 1))
            continue
        fi

        PORT=$((BASE_PORT + i - 1))

        # Try the health check
        if curl -s -f --connect-timeout 2 --max-time 5 http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
            READY_WORKERS=$((READY_WORKERS + 1))
            WORKER_READY[$i]=1
            echo "  ✅ Worker $i (port $PORT): Ready! ($READY_WORKERS/$NUM_WORKERS)"
        fi
    done

    # Show progress every 10 seconds
    if [ $((CURRENT_TIME % 10)) -eq 0 ] && [ $READY_WORKERS -lt $NUM_WORKERS ]; then
        echo "  ⏳ Progress: $READY_WORKERS/$NUM_WORKERS workers ready (${CURRENT_TIME}s elapsed)"
    fi

    # Check less frequently to reduce CPU usage and log spam
    sleep 2
done

echo "All workers are ready!"

# Start nginx
echo "Starting nginx on port $NGINX_PORT..."
nginx

echo "=== Multi-worker deployment ready ==="
echo "🚀 Nginx load balancer: http://localhost:$NGINX_PORT"
echo "📊 Session affinity: enabled (based on X-Session-ID header or JSON session_id)"
echo "⚙️  Workers: $NUM_WORKERS (ports $BASE_PORT-$((BASE_PORT + NUM_WORKERS - 1)))"
echo "📈 Nginx status: http://localhost:$NGINX_PORT/nginx-status"
echo "📊 UWSGI processes per worker: $UWSGI_PROCESSES"
if [ -n "$UWSGI_CHEAPER" ]; then
    echo "💰 UWSGI cheaper mode: $UWSGI_CHEAPER"
else
    echo "💰 UWSGI cheaper mode: disabled"
fi

# Show process status
echo "📊 Process status:"
for i in "${!WORKER_PIDS[@]}"; do
    pid=${WORKER_PIDS[$i]}
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Worker $((i+1)) (PID $pid): Running"
    else
        echo "  Worker $((i+1)) (PID $pid): Dead"
    fi
done

# Keep the container running and monitor
echo "Monitoring processes (Ctrl+C to stop)..."
while true; do
    # Check if any worker died
    for i in "${!WORKER_PIDS[@]}"; do
        pid=${WORKER_PIDS[$i]}
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: Worker $((i+1)) (PID $pid) died unexpectedly"
            echo "Worker log:"
            tail -20 /var/log/worker$((i+1)).log 2>/dev/null || echo "No log found"
            cleanup
            exit 1
        fi
    done

    # Check nginx
    if ! pgrep nginx > /dev/null; then
        echo "ERROR: Nginx died unexpectedly"
        cleanup
        exit 1
    fi

    sleep 10
done
