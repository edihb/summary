---
layout: post
title: '"Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning" Summary'
---

<style>
{% include blogposts.css %}
</style>

Information extraction (IE) is the task of automatically extracting structured information from unstructured or semi-structured machine readable documents. IE systems require large amounts of annotated data to deliver high performance. This paper summary describes the task of acquiring and incorporating <b>external evidence</b> that improves extraction accuracy when <b>training data is scarce</b>. 

## Databases used in the paper 
1. Shooting incidents
2. Food adulteration cases

## Sample IE system
### The problem

![fig1](/images/sample_news_article.png){: .center-image }
<center><b>Figure 1: Sample news article on a shooting case.</b></center>
<br>
The document does not explicitly name the shooter (<i>Scott Westerhuis</i>). The total number of fatalities is also not explicitly metioned (<i>A couple and four children</i><b>=6</b>) <br>
This paper suggests that a large annotated training set may not cover all such cases and proposes a system that utilizes <b>external sources</b> to resolve such text interpretation ambiguities.
### The solution
![fig2](/images/2articles.png){: .center-image }
<center><b>Figure 2: Articles with explicit mentions of fatality count and shooter name.</b></center>
<br>
The new IE system involves querying for such alternative articles and extracting information from them instead of the original source. <br>
Such stereotypical phrasing is easier to process for most IE systems and the new system boosts extraction accuracy compared to traditional extractor systems by upto 11.5%.<br>
### Challenges with the new system <br>
1. <b>Performing event coreference -</b><br>
	Retrieving suitable articles describing the same event.
2. <b>Reconciling the entities extracted from different documents obtained -</b><br>
	Sometimes tangential results are obtained as querying for similar documents returns documents about other similar incidents.<br>
	Additionally, values extracted from different sources may be different (as some sources are inaccurate).<br>
	
## Framework
<b>Markov Decision Process(MDP)</b> is introduced to tackle the challenge of reconciling extracted entities. The system goes through alternating phases of querying to retrieve the articles and integrating the extracted values if the confidence is sufficiently high. It satisfies the <i>Markov Property</i> as the future state of the model depends only on the present state and not on the sequence of all preceeding states.<br>
<b>A Reinforcement Learning(RL)</b> approach is is used to combine query formulation(for event coreference), extraction from new sources and value reconciliation.
<br>
![rl](/images/reinforcement_learning.png){: .floatright}

The model selects good actions for both article retrieval and value reconciliation(<b>action</b>) in order to optimize the <b>reward function</b> that reflects extraction accuracy and penalties for extra moves.<br>
<b>RL agent</b> is trained using a [Deep Q-Network (DQN)](https://deepmind.com/research/dqn/) that simultaneously predicts both querying and reconciliation choices.<br>
Maximum entropy model is used as the base extractor for <b>interpretation</b>.
<br>

## Information extraction task as a Markov Decision Process (MDP)
Introducing the MDP framework is important as it allows dynamic integration of entity predictions while simultaneously providing flexibility to choose the type of articles to extract from (query template).<br>

![mdp](/images/mdp_transition.png){: .center-image }
<center><b>Figure 3: Illustration of transition in MDP.</b></center>
<br>
The top box of each state depics the current entities and the bottom box shows new entities extracted from a downloaded article on the same event after querying.<br>
At each step the MDP decides whether to accept an entities value and reconcile it or continue inspecting further articles by generating queries from a template created using the title of the article along with words most likely to co-occur with each entity type.<br>
Since the IR System is looking at the shooter database, the entities extracted are as follows -<br>
![entities](/images/entity_values.png){: .center-image }
<br>
<b>MDP as a tuple: (S, A, T, R)</b><br>
Where, <br>
<i><b>S = {s}</b></i> is the state space.<br>
![states](/images/state_rep.png){: .center-image }
<br>
The state of the MDP is represented based on the values of the entities.<br>
The state representation depends on the following -<br>
1. Current confidence and new confidence of entities.
2. One hot encoding matches between current and new values.
3. Unigram/tf-idf counts of context words.
4. tf-idf similarity between original and new article.
<br><br>

<i><b>A = {a=(d,q)}</b></i> is the set of all actions depending on decision d and query choice q.<br>
At each step, the agent makes a reconciliation decision <i>d</i> and query choice <i>q</i> with the episode ending when all entity values are accepted and the stop action is chosen.<br><br>
<i><b>R(s,a)</b></i> is the reward function<br>
![reward](/images/reward_function.png){: .center-image }
<br>
The reward function maximizes the final extraction accuracy while minimizing the number of queries by setting a negative reward to penalize the agent for longer episodes.<br><br>
<i><b>T(s'|s,a)</b></i> is the transition function<br>
The transition funciton maps the old state to the new state based on the action chosen by the agent.<br>

## Reinforcement Learning for Information Extraction
The agent employs a <i><b>learning function Q(s,a)</b></i> to determine which action to perform in state <i>s</i>.<br>
The learning technique employed is [Q-learning](https://link.springer.com/article/10.1007/BF00992698) wherin the agent iteratively updates Q(s,a) using the [recursive Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) and rewards function <i>r</i>.<br>
![bellman](/images/bellman.png){: .center-image }
<br>
Thus we can see that the dynamic optimization problem is broken down into smaller sub-problems involving the reward <i>r</i> and future rewards over all possible transitions discounted by a factor γ.<br>
Since the state space is continuous, a [deep Q-Network (DQN)](https://deepmind.com/research/dqn/) is used as a function approximator. It can continuously adapt is behaviour without any human intervention to capture non-linear interactions  between information in the states and performs better than linear approximators.<br>
The DQN used consists of two linear layers (20 hidden units each) followed by rectified linear units (ReLU), along with two separate output layers to simultaneously predict <i><b>Q(s,d)</b></i> for reconciliation decisions and <i><b>Q(s,q)</b></i> for query choices.<br>
<br>
![rlalgo](/images/algorithm.png){: .center-image }
<br>
<center><b>Figure 4: DQN training procedure algorith.</b></center>
The above algorithm details the DQN training procedure.<br>
Just as in any other ML algorithm, the loss function is being minimized.<br>
Stochastic gradient descent with RMSprop is used for <b>Parameter Learning</b> of parameters θ of the DQN.<br>
A breif description the optimization methods (RMSprop and more) for Deep Networks can be found [here](http://www.cs.cmu.edu/~imisra/data/Optimization_2015_11_11.pdf).<br>
Each parameter update aims to close the gap between the Q(st,at; θ) predicted by the DQN and the expected Q-value from the Bellman equation.<br>
## Experiment
### Data
Annotated datasets with news articles and entities to be extracted.<br>
### Extraction model
Bing Search API is used for different automatically generated queries from the template.<br><br>
<b>Baseline classifiers</b>
1. <b>Basic Extractors -</b><br>
	* Maximum Entropy 
	* CRF (results were worse than Maxent)
2. <b>Aggregation Systems -</b><br>
	* Confidence model for value reconciliation that selects the entity with the highest confidence core assigned by the base extractor.
	* Majority model that takes the majority vote over all extracted values.
3. <b>Meta-classifier -</b><br> 
	Demonstrates the importance of modelling the problem in RL Framework.<br>
	This classifier does not implement a classification algorithm of its own but operates over the same input space as DQN and produces the same set of reconciliation decisions {d} by aggregating value predictions using the confidence based scheme.
4. <b>Oracle -</b><br>
	Gold standard score computed assuming perfect reconciliation and querying decisions on top of Maxent base extractor to analyze the contributions of the RL system in isolation of base extractor limitations.<br>
### RL Models
* <b>RL-Basic -</b><br>
	Performs only reconciliation decisions.
* <b>RL-Query -</b><br>
	Takes only query decisions with reconciliation strategy fixed.
* <b>RL-Extract -</b><br>
	Full system incorporating both reconciliation and query decisions.<br><br>

## Results
![results](/images/results.png){: .center-image }
<center><b>Figure 4: Accuracy of baseline, DQN and Oracle on Shootings and Adulteration datasets.</b></center>
<br>
The above table shows that the new system (RL-Extract) obtains a substantial gain over all the baseline classifiers over both domains.<br>
The importance of sequential decision making can be seen as RL-Extract performs significantly better than the meta-classifier. This is because the meta-classifier aggregates all documents including noisy and irrelevant documents.<br>
Enabling RL-Query results in significant improvement over RL-Basic in both domains demonstrating the need to perform query selection and reconciliation simultaneously.	
	
	
## Conclusion	
![con](/images/conclusion.png){: .center-image }
<center><b>Figure 5: Evolution of average reward (solid black) and accuracy on various entities (red=ShooterName, magenta=NumKilled, blue=NumWounded, green=City) on the Shootings domain test set.</b></center>
<br>	
The above learning curves show that the reward improves gradually and the accuracy on the entity increases simultaneously with each iteration.<br>
Thus in domains with scarcity of training data, the new system is found to show considerable improvement over existing baseline in terms of accuracy of extraction.<br>	
		
## Acknowledgements
<cite> 
Narasimhan, Karthik, Adam Yala, and Regina Barzilay. "Improving Information Extraction by Acquiring External Evidence with Reinforcement Learning." Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016. doi:10.18653/v1/d16-1261.
</cite>
<br> <br>
https://github.com/karthikncode/DeepRL-InformationExtraction





