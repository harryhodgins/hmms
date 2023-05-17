import numpy as np
import random
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from hmmlearn.hmm import CategoricalHMM
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


#1 beats 2,3 -- loses to 4,5
#2 beats 3,4 -- loses to 5,1
#3 beats 4,5 -- loses to 1,2
#4 beats 5,1 -- loses to 2,3
#5 beats 1,2 -- loses to 3,4

population = [1,2,3,4,5]

#possible_outcomes = random.choices(population,weights = (20,20,20,20,20),k = 10000)
possible_outcomes = []
for i in range(2000):
    possible_outcomes.append(1)
    possible_outcomes.append(1)
    possible_outcomes.append(4)
    possible_outcomes.append(3)
    possible_outcomes.append(2)
    
list_1 = [4,5]
list_2 = [1,5]
list_3 = [1,2]
list_4 = [2,3]
list_5 = [3,4]

dataframe = pd.DataFrame({'choice' : possible_outcomes})


train_size = int(0.9 * len(dataframe))
train_data = dataframe.iloc[0:train_size]
model = GaussianHMM(n_components=5) 
#model = CategoricalHMM(n_components=5)    
model.fit(train_data)    
test_data = dataframe.iloc[train_size+1:]

#bad_test = random.choices(population,weights = (100,0,0,0,0),k = len(dataframe) - train_size-1)

#bad_player = pd.DataFrame({'bad choice' : bad_test})
#test_data = bad_player
latent_guesses = 10
num_moves_predict = 100
possibles = [1,2,3,4,5]
#possibles = np.linspace(1,5,100)
guesses = []
wins = 0
loses = 0
ties = 0
computer_choice = []
bank = []

for i in tqdm(range(num_moves_predict)):
    start_index = max(0,i - latent_guesses)
    end_index = max(0,i)
    
    previous_data = test_data.iloc[start_index:end_index]
    outcome_scores = []

    for outcome in possibles:
        total_data = np.row_stack((previous_data,outcome))
        outcome_scores.append(model.score(total_data))
    
    most_probable_outcome = possibles[np.argmax(outcome_scores)]
    guesses.append(most_probable_outcome)
    
    for i in range(len(guesses)):
        guesses[i] = math.floor(guesses[i])
        
    
    if guesses[i] == 1:
        computer_choice.append(random.choice(list_1))
    elif guesses[i] ==2:
        computer_choice.append(random.choice(list_2))
    elif guesses[i] ==3:
        computer_choice.append(random.choice(list_3))
    elif guesses[i] ==4:
        computer_choice.append(random.choice(list_4))
    elif guesses[i] ==5:
        computer_choice.append(random.choice(list_5))
        
    
for i in range(len(computer_choice)):
    if computer_choice[i] == 1:
        if test_data.iloc[i][0] == 2 or test_data.iloc[i][0] == 3:
            wins += 1
        elif test_data.iloc[i][0] == 1:
            ties += 1
        else: 
            loses += 1
            
        
    if computer_choice[i] == 2:
        if test_data.iloc[i][0] == 3 or test_data.iloc[i][0] == 4:
            wins += 1
        elif test_data.iloc[i][0] == 2:
            ties += 1
        else: 
            loses += 1
            
    if computer_choice[i] == 3:
        if test_data.iloc[i][0] == 4 or test_data.iloc[i][0] == 5:
            wins += 1
        elif test_data.iloc[i][0] == 3:
            ties += 1
        else: 
            loses += 1
            
    if computer_choice[i] == 4:
        if test_data.iloc[i][0] == 5 or test_data.iloc[i][0] == 1:
            wins += 1
        elif test_data.iloc[i][0] == 4:
            ties += 1
        else: 
            loses += 1
            
    if computer_choice[i] == 5:
        if test_data.iloc[i][0] == 1 or test_data.iloc[i][0] == 2:
            wins += 1
        elif test_data.iloc[i][0] == 5:
            ties += 1
        else: 
            loses += 1
        
    
plt.figure(figsize=(20,10), dpi=80)
x_axis = np.array(test_data.index[0:(num_moves_predict)])
plt.plot(x_axis,test_data.iloc[0:num_moves_predict], 'b+-', label="Actual")
plt.plot(x_axis, guesses, 'ro-', label="Predicted")
plt.legend(fontsize = 16)
plt.title('Actual vs. Predicted Moves',fontsize = 30)
plt.xlabel('Moves', fontsize = 24)
plt.ylabel('Card Choice',fontsize = 24)
plt.show()

print(wins)
print(loses)
print(ties)

win_percentage = wins / num_moves_predict
loss_percentage = loses / num_moves_predict
tie_percentage = ties / num_moves_predict

# labels = ['win','lose','tie']
# sizes = [win_percentage,loss_percentage,tie_percentage]

# fig,ax = plt.subplots()
# ax.pie(sizes,labels= labels)


labels = ['win','lose','tie']
sizes = [win_percentage,loss_percentage,tie_percentage]
#explode = (0.1, 0, 0)

#fig,ax = plt.subplots()
#ax.pie(sizes,labels= labels)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Results")

