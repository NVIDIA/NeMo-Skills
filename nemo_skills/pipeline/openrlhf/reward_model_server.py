
import multiprocessing

class RewardModelProxy:
    def __init__(self, test_limit: int):
        self.test_limit = test_limit

    def get_reward(self, queries, input_dicts):
        test_data = [(seq, input_dict['solution'], self.test_limit, execute) for seq, input_dict in zip(queries, input_dicts)]
        with multiprocessing.Pool(processes=len(queries)) as pool:
            rewards = pool.starmap(evaluate_test_case, tqdm(test_data, total=len(queries)))
        return rewards
