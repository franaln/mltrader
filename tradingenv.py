#
import gym
from gym import spaces

HOLD = 0
BUY  = 1
SELL = 2

class TradingEnv(gym.Env):

    """A trading environment for OpenAI gym"""

    metadata = {'render.modes': ['human', 'system', 'none']}

    def __init__(self, env_config):
        super(TradingEnv, self).__init__()

        self.data = env_config['data']

        self.features = env_config.get('features')
        self.window_size = env_config.get('window_size', 8)
        self.simulation_length = env_config.get('simulation_length', 168)

        self.stake_amount = env_config.get('stake_amount', 100.)
        self.init_mode = env_config.get('init_mode', 'random')
        self.current_step = 0

        self.trade = None

        self.total_reward = 0
        self.current_tick = 0

        # Actions
        self.action_space = spaces.Discrete(3)

        # Observations
        self.shape = (self.window_size*len(self.features)+1,)
        self.obs = np.empty(self.shape)
        #self.observation_space = spaces.Box(low=-10, high=10, shape=self.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)


    def _get_observation(self):

        # trade status
        trade_status = 1 if self.trade is not None else 0
        self.obs[0] = trade_status

        # data
        self.obs[1:] = self.data.iloc[self.current_tick-self.window_size+1:self.current_tick+1][self.features].to_numpy().ravel()

        return self.obs


    def _take_action(self, action):
        """
        take action (HOLD, BUY or SELL) and return reward
        """

        buy_cost  = -0.001
        sell_cost = -0.001

        if action == HOLD:
            return 0

        elif action == BUY:

            # if already in trade do nothing and return 0
            if self.trade is not None:
                return buy_cost

            # create trade
            amount = self.stake_amount / self.current_price

            self.trade = {
                'type': 'buy',
                'open_rate': self.current_price,
                'open_date': self.current_date,
                'amount': amount,
                'is_open': True,
            }

            return buy_cost

        elif action == SELL:

            # if no trade do nothing and return 0
            if self.trade is None:
                return sell_cost

            profit =  self.trade['amount'] * (self.current_price - self.trade['open_rate'])

            self.trade = None

            return profit + sell_cost


    def step(self, action):

        # Execute one time step within the environment
        done = False

        self.steps += 1
        self.current_tick += 1

        if self.current_tick >= (len(self.data) - self.simulation_length - 1):
            self.current_step = 0
            return np.zeros(self.shape), 0, True, {}

        if self.steps > self.simulation_length:
            return np.zeros(self.shape), 0, True, {}


        # current status
        self.current_date   = self.data.iloc[self.current_tick].name
        self.current_price  = self.data.iloc[self.current_tick]['open']

        info = {'date': self.current_date}

        step_reward = self._take_action(action)

        observation = self._get_observation()

        self.total_reward += step_reward

        return observation, step_reward, False, info


    def reset(self):

        # Reset the state of the environment to an initial state
        self.steps = 0
        self.trade = None
        self.trades = []
        self.total_reward = 0

        if self.init_mode == 'random':
            self.current_tick = random.randint(self.window_size+1, len(self.data)-self.simulation_length-self.window_size-1)
        elif self.init_mode == 'walk':
            self.current_tick = (self.window_size + 1) + self.simulation_length * self.current_step
        elif self.init_mode == 'same':
            self.current_tick = self.window_size + 1

        self.current_step += 1

        return self._get_observation()


    def render(self, mode='live', close=False):
        # Render the environment to the screen
        print(f'Step: {self.steps}, Reward: {self.total_reward}')


    def print_summary(self):
        print(f'Trades: {len(self.trades)}, Total reward: {self.total_reward:.2f} BUSD')
