<!--
 * @Description: Some papers you might want to use
 * @version: V1.0
 * @Author: lesheng
 * @Date: 2021-04-15 18:50:20
 * @LastEditors: lesheng
 * @LastEditTime: 2021-04-16 14:39:26
-->

# 深度强化学习核心论文

- [深度强化学习核心论文](#深度强化学习核心论文)
  - [免模型强化学习](#免模型强化学习)
    - [深度 Q-Learning](#深度-q-learning)
    - [策略梯度](#策略梯度)
    - [确定性策略梯度](#确定性策略梯度)
    - [分布式强化学习](#分布式强化学习)
    - [带有 Action-Dependent Baselines 的策略梯度](#带有-action-dependent-baselines-的策略梯度)
    - [路径一致性学习（Path-Consistency Learning）](#路径一致性学习path-consistency-learning)
    - [其他结合策略梯度和Q-Learning的方向](#其他结合策略梯度和q-learning的方向)
    - [进化算法](#进化算法)
  - [探索](#探索)
    - [内在激励（Intrinsic Motivation）](#内在激励intrinsic-motivation)
    - [非监督强化学习](#非监督强化学习)
  - [迁移和多任务强化学习](#迁移和多任务强化学习)
  - [层次（Hierarchy）](#层次hierarchy)
  - [记忆（Memory）](#记忆memory)
  - [有模型强化学习](#有模型强化学习)
    - [模型可被学习](#模型可被学习)
    - [模型已知](#模型已知)
  - [元学习（Meta-RL）](#元学习meta-rl)
  - [扩展强化学习](#扩展强化学习)
  - [现实世界的强化学习](#现实世界的强化学习)
  - [安全性](#安全性)
  - [模仿学习和逆强化学习](#模仿学习和逆强化学习)
  - [可复现、分析和评价](#可复现分析和评价)
  - [强化学习理论的经典论文](#强化学习理论的经典论文)

## 免模型强化学习

### 深度 Q-Learning

- [1][Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)__Playing Atari with Deep Reinforcement Learning, Mnih et al, 2013. Algorithm: DQN.__
- [2][Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)__Deep Recurrent Q-Learning for Partially Observable MDPs, Hausknecht and Stone, 2015. Algorithm: Deep Recurrent Q-Learning.__
- [3][Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)__Dueling Network Architectures for Deep Reinforcement Learning, Wang et al, 2015. Algorithm: Dueling DQN.__  
- [4][Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)__Deep Reinforcement Learning with Double Q-learning, Hasselt et al 2015. Algorithm: Double DQN.__  
- [5][Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)__Prioritized Experience Replay, Schaul et al, 2015. Algorithm: Prioritized Experience Replay (PER).__  
- [6][Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)__Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et al, 2017. Algorithm: Rainbow DQN.__  

### 策略梯度

- [7][Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)__Asynchronous Methods for Deep Reinforcement Learning, Mnih et al, 2016. Algorithm: A3C.__  
- [8][Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)__Trust Region Policy Optimization, Schulman et al, 2015. Algorithm: TRPO.__  
- [9][High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)__High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al, 2015. Algorithm: GAE.__  
- [10][Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)__Proximal Policy Optimization Algorithms, Schulman et al, 2017. Algorithm: PPO-Clip, PPO-Penalty.__  
- [11][Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)__Emergence of Locomotion Behaviours in Rich Environments, Heess et al, 2017. Algorithm: PPO-Penalty.__  
- [12][Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)__Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation, Wu et al, 2017. Algorithm: ACKTR.__  
- [13][Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224)__Sample Efficient Actor-Critic with Experience Replay, Wang et al, 2016. Algorithm: ACER.__  
- [14][Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)__Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor, Haarnoja et al, 2018. Algorithm: SAC.__  

### 确定性策略梯度

- [15][Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)__Deterministic Policy Gradient Algorithms, Silver et al, 2014. Algorithm: DPG.__  
- [16][Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)__Continuous Control With Deep Reinforcement Learning, Lillicrap et al, 2015. Algorithm: DDPG.__  
- [17][Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)__Addressing Function Approximation Error in Actor-Critic Methods, Fujimoto et al, 2018. Algorithm: TD3.__  

### 分布式强化学习

- [18][A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)__A Distributional Perspective on Reinforcement Learning, Bellemare et al, 2017. Algorithm: C51.__  
- [19][Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)__Distributional Reinforcement Learning with Quantile Regression, Dabney et al, 2017. Algorithm: QR-DQN.__  
- [20][Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)__Implicit Quantile Networks for Distributional Reinforcement Learning, Dabney et al, 2018. Algorithm: IQN.__  
- [21][Dopamine: A Research Framework for Deep Reinforcement Learning](https://openreview.net/forum?id=ByG_3s09KX)__Dopamine: A Research Framework for Deep Reinforcement Learning, Anonymous, 2018. Contribution: Introduces Dopamine, a code repository containing implementations of DQN, C51, IQN, and Rainbow. Code link.__  

### 带有 Action-Dependent Baselines 的策略梯度

- [22][Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247)__Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic, Gu et al, 2016. Algorithm: Q-Prop.__  
- [23][Action-depedent Control Variates for Policy Optimization via Stein’s Identity](https://arxiv.org/abs/1710.11198)__Action-depedent Control Variates for Policy Optimization via Stein’s Identity, Liu et al, 2017. Algorithm: Stein Control Variates.__  
- [24][The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://arxiv.org/abs/1802.10031)__The Mirage of Action-Dependent Baselines in Reinforcement Learning, Tucker et al, 2018. Contribution: interestingly, critiques and reevaluates claims from earlier papers (including Q-Prop and stein control variates) and finds important methodological errors in them.__  

### 路径一致性学习（Path-Consistency Learning）

- [25][Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)__Bridging the Gap Between Value and Policy Based Reinforcement Learning, Nachum et al, 2017. Algorithm: PCL.__  
- [26][Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891)__Trust-PCL: An Off-Policy Trust Region Method for Continuous Control, Nachum et al, 2017. Algorithm: Trust-PCL.__  

### 其他结合策略梯度和Q-Learning的方向

- [27][Combining Policy Gradient and Q-learning](https://arxiv.org/abs/1611.01626)__Combining Policy Gradient and Q-learning, O’Donoghue et al, 2016. Algorithm: PGQL.__  
- [28][The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning](https://arxiv.org/abs/1704.04651)__The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning, Gruslys et al, 2017. Algorithm: Reactor.__  
- [29][Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning)__Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning, Gu et al, 2017. Algorithm: IPG.__  
- [30][Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)__Equivalence Between Policy Gradients and Soft Q-Learning, Schulman et al, 2017. Contribution: Reveals a theoretical link between these two families of RL algorithms.__  

### 进化算法

- [31][Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)__Evolution Strategies as a Scalable Alternative to Reinforcement Learning, Salimans et al, 2017. Algorithm: ES.__  

## 探索

### 内在激励（Intrinsic Motivation）

- [32][VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674)__VIME: Variational Information Maximizing Exploration, Houthooft et al, 2016. Algorithm: VIME.__  
- [33][Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)__Unifying Count-Based Exploration and Intrinsic Motivation, Bellemare et al, 2016. Algorithm: CTS-based Pseudocounts.__  
- [34][Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)__Count-Based Exploration with Neural Density Models, Ostrovski et al, 2017. Algorithm: PixelCNN-based Pseudocounts.__  
- [35][#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717)__#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning, Tang et al, 2016. Algorithm: Hash-based Counts.__  
- [36][EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260)__EX2: Exploration with Exemplar Models for Deep Reinforcement Learning, Fu et al, 2017. Algorithm: EX2.__  
- [37][Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)__Curiosity-driven Exploration by Self-supervised Prediction, Pathak et al, 2017. Algorithm: Intrinsic Curiosity Module (ICM).__  
- [38][Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)__Large-Scale Study of Curiosity-Driven Learning, Burda et al, 2018. Contribution: Systematic analysis of how surprisal-based intrinsic motivation performs in a wide variety of environments.__  
- [39][Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)__Exploration by Random Network Distillation, Burda et al, 2018. Algorithm: RND.__  

### 非监督强化学习

- [40][Variational Intrinsic Control](https://arxiv.org/abs/1611.07507)__Variational Intrinsic Control, Gregor et al, 2016. Algorithm: VIC.__  
- [41][Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070)__Diversity is All You Need: Learning Skills without a Reward Function, Eysenbach et al, 2018. Algorithm: DIAYN.__  
- [42][Variational Option Discovery Algorithms](https://arxiv.org/abs/1807.10299)__Variational Option Discovery Algorithms, Achiam et al, 2018. Algorithm: VALOR.__  

## 迁移和多任务强化学习

- [43][Progressive Neural Networks](https://arxiv.org/abs/1606.04671)__Progressive Neural Networks, Rusu et al, 2016. Algorithm: Progressive Networks.__  
- [44][Universal Value Function Approximators](http://proceedings.mlr.press/v37/schaul15.pdf)__Universal Value Function Approximators, Schaul et al, 2015. Algorithm: UVFA.__  
- [45][Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)__Reinforcement Learning with Unsupervised Auxiliary Tasks, Jaderberg et al, 2016. Algorithm: UNREAL.__  
- [46][The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300)__The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously, Cabi et al, 2017. Algorithm: IU Agent.__  
- [47][PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/abs/1701.08734)__PathNet: Evolution Channels Gradient Descent in Super Neural Networks, Fernando et al, 2017. Algorithm: PathNet.__  
- [48][Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907)__Mutual Alignment Transfer Learning, Wulfmeier et al, 2017. Algorithm: MATL.__  
- [49][Learning an Embedding Space for Transferable Robot Skills](https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb)__Learning an Embedding Space for Transferable Robot Skills, Hausman et al, 2018.__  
- [50][Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)__Hindsight Experience Replay, Andrychowicz et al, 2017. Algorithm: Hindsight Experience Replay (HER).__  

## 层次（Hierarchy）

- [51][Strategic Attentive Writer for Learning Macro-Actions](https://arxiv.org/abs/1606.04695)__Strategic Attentive Writer for Learning Macro-Actions, Vezhnevets et al, 2016. Algorithm: STRAW.__  
- [52][FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161)__FeUdal Networks for Hierarchical Reinforcement Learning, Vezhnevets et al, 2017. Algorithm: Feudal Networks__  
- [53][Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)__Data-Efficient Hierarchical Reinforcement Learning, Nachum et al, 2018. Algorithm: HIRO.__  

## 记忆（Memory）

- [54][Model-Free Episodic Control](https://arxiv.org/abs/1606.04460)__Model-Free Episodic Control, Blundell et al, 2016. Algorithm: MFEC.__  
- [55][Neural Episodic Control](https://arxiv.org/abs/1703.01988)__Neural Episodic Control, Pritzel et al, 2017. Algorithm: NEC.__  
- [56][Neural Map: Structured Memory for Deep Reinforcement Learning](https://arxiv.org/abs/1702.08360)__Neural Map: Structured Memory for Deep Reinforcement Learning, Parisotto and Salakhutdinov, 2017. Algorithm: Neural Map.__  
- [57][Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/abs/1803.10760)__Unsupervised Predictive Memory in a Goal-Directed Agent, Wayne et al, 2018. Algorithm: MERLIN.__  
- [58][Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822)__Relational Recurrent Neural Networks, Santoro et al, 2018. Algorithm: RMC.__  

## 有模型强化学习

### 模型可被学习

- [59][Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203)__Imagination-Augmented Agents for Deep Reinforcement Learning, Weber et al, 2017. Algorithm: I2A.__  
- [60][Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596)__Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning, Nagabandi et al, 2017. Algorithm: MBMF.__  
- [61][Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/abs/1803.00101)__Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning, Feinberg et al, 2018. Algorithm: MVE.__  
- [62][Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675)__Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion, Buckman et al, 2018. Algorithm: STEVE.__  
- [63][Model-Ensemble Trust-Region Policy Optimization](https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ)__Model-Ensemble Trust-Region Policy Optimization, Kurutach et al, 2018. Algorithm: ME-TRPO.__  
- [64][Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214)__Model-Based Reinforcement Learning via Meta-Policy Optimization, Clavera et al, 2018. Algorithm: MB-MPO.__  
- [65][Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999)__Recurrent World Models Facilitate Policy Evolution, Ha and Schmidhuber, 2018.__  

### 模型已知

- [66][Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)__Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm, Silver et al, 2017. Algorithm: AlphaZero.__  
- [67][Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439)__Thinking Fast and Slow with Deep Learning and Tree Search, Anthony et al, 2017. Algorithm: ExIt.__  

## 元学习（Meta-RL）

- [68][RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)__RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning, Duan et al, 2016. Algorithm: RL^2.__  
- [69][Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763)__Learning to Reinforcement Learn, Wang et al, 2016.__  
- [70][Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)__Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, Finn et al, 2017. Algorithm: MAML.__  
- [71][A Simple Neural Attentive Meta-Learner](https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW)__A Simple Neural Attentive Meta-Learner, Mishra et al, 2018. Algorithm: SNAIL.__  

## 扩展强化学习

- [72][Accelerated Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1803.02811)__Accelerated Methods for Deep Reinforcement Learning, Stooke and Abbeel, 2018. Contribution: Systematic analysis of parallelization in deep RL across algorithms.__  
- [73][IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)__IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures, Espeholt et al, 2018. Algorithm: IMPALA.__  
- [74][Distributed Prioritized Experience Replay](https://openreview.net/forum?id=H1Dy---0Z)__Distributed Prioritized Experience Replay, Horgan et al, 2018. Algorithm: Ape-X.__  
- [75][Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)__Recurrent Experience Replay in Distributed Reinforcement Learning, Anonymous, 2018. Algorithm: R2D2.__  
- [76][RLlib: Abstractions for Distributed Reinforcement Learning](https://arxiv.org/abs/1712.09381)__RLlib: Abstractions for Distributed Reinforcement Learning, Liang et al, 2017. Contribution: A scalable library of RL algorithm implementations. Documentation link.__  

## 现实世界的强化学习

- [77][Benchmarking Reinforcement Learning Algorithms on Real-World Robots](https://arxiv.org/abs/1809.07731)__Benchmarking Reinforcement Learning Algorithms on Real-World Robots, Mahmood et al, 2018.__  
- [78][Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177)__Learning Dexterous In-Hand Manipulation, OpenAI, 2018.__  
- [79][QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation](https://arxiv.org/abs/1806.10293)__QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation, Kalashnikov et al, 2018. Algorithm: QT-Opt.__  
- [80][Horizon: Facebook’s Open Source Applied Reinforcement Learning Platform](https://arxiv.org/abs/1811.00260)__Horizon: Facebook’s Open Source Applied Reinforcement Learning Platform, Gauci et al, 2018.__  

## 安全性

- [81][Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)__Concrete Problems in AI Safety, Amodei et al, 2016. Contribution: establishes a taxonomy of safety problems, serving as an important jumping-off point for future research. We need to solve these!__  
- [82][Deep Reinforcement Learning From Human Preferences](https://arxiv.org/abs/1706.03741)__Deep Reinforcement Learning From Human Preferences, Christiano et al, 2017. Algorithm: LFP.__  
- [83][Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)__Constrained Policy Optimization, Achiam et al, 2017. Algorithm: CPO.__  
- [84][Safe Exploration in Continuous Action Spaces](https://arxiv.org/abs/1801.08757)__Safe Exploration in Continuous Action Spaces, Dalal et al, 2018. Algorithm: DDPG+Safety Layer.__  
- [85][Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)__Trial without Error: Towards Safe Reinforcement Learning via Human Intervention, Saunders et al, 2017. Algorithm: HIRL.__  
- [86][Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning](https://arxiv.org/abs/1711.06782)__Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning, Eysenbach et al, 2017. Algorithm: Leave No Trace.__  

## 模仿学习和逆强化学习

- [87][Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy](http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)__Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy, Ziebart 2010. Contributions: Crisp formulation of maximum entropy IRL.__  
- [88][Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization](https://arxiv.org/abs/1603.00448)__Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization, Finn et al, 2016. Algorithm: GCL.__  
- [89][Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)__Generative Adversarial Imitation Learning, Ho and Ermon, 2016. Algorithm: GAIL.__  
- [90][DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf)__DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, Peng et al, 2018. Algorithm: DeepMimic.__  
- [91][Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/abs/1810.00821)__Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow, Peng et al, 2018. Algorithm: VAIL.__  
- [92][One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL](https://arxiv.org/abs/1810.05017)__One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL, Le Paine et al, 2018. Algorithm: MetaMimic.__  

## 可复现、分析和评价

- [93][Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778)__Benchmarking Deep Reinforcement Learning for Continuous Control, Duan et al, 2016. Contribution: rllab.__  
- [94][Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133)__Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control, Islam et al, 2017.__  
- [95][Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)__Deep Reinforcement Learning that Matters, Henderson et al, 2017.__  
- [96][Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods](https://arxiv.org/abs/1810.02525)__Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods, Henderson et al, 2018.__  
- [97][Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?](https://arxiv.org/abs/1811.02553)__Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms?, Ilyas et al, 2018.__  
- [98][Simple Random Search Provides a Competitive Approach to Reinforcement Learning](https://arxiv.org/abs/1803.07055)__Simple Random Search Provides a Competitive Approach to Reinforcement Learning, Mania et al, 2018.__  
- [99][Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/abs/1907.02057)__Benchmarking Model-Based Reinforcement Learning, Wang et al, 2019.__  

## 强化学习理论的经典论文

- [100][Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)__Policy Gradient Methods for Reinforcement Learning with Function Approximation, Sutton et al, 2000. Contributions: Established policy gradient theorem and showed convergence of policy gradient algorithm for arbitrary policy classes.__  
- [101][An Analysis of Temporal-Difference Learning with Function Approximation](http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)__An Analysis of Temporal-Difference Learning with Function Approximation, Tsitsiklis and Van Roy, 1997. Contributions: Variety of convergence results and counter-examples for value-learning methods in RL.__  
- [102][Reinforcement Learning of Motor Skills with Policy Gradients](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf)__Reinforcement Learning of Motor Skills with Policy Gradients, Peters and Schaal, 2008. Contributions: Thorough review of policy gradient methods at the time, many of which are still serviceable descriptions of deep RL methods.__  
- [103][Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)__Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford, 2002. Contributions: Early roots for monotonic improvement theory, later leading to theoretical justification for TRPO and other algorithms.__  
- [104][A Natural Policy Gradient](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)__A Natural Policy Gradient, Kakade, 2002. Contributions: Brought natural gradients into RL, later leading to TRPO, ACKTR, and several other methods in deep RL.__  
- [105][Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)__Algorithms for Reinforcement Learning, Szepesvari, 2009. Contributions: Unbeatable reference on RL before deep RL, containing foundations and theoretical background.__  
