import numpy as np
from matplotlib import pyplot as plt 
from owlready2 import *
import random


class Agent():
    def __init__(self, ontology, lr, gamma, p, epsilon_parameters = (0.5, 0.5, 0.1, 10)):
        self.states = ontology.Entity.instances()
        self.actions = ontology.Action.instances()
        self.Q_table = np.zeros((len(self.states), len(self.actions)))
        self.senses = ontology.InternalSense.instances()
        # Hyperparameters : 
        self.lr = lr 
        self.gamma = gamma 
        self.p = p # paramètre pour calculer la distance d'édition
        self.esp_param = epsilon_parameters 
    
    # Fonctions pour mettre à jour la Q-table avec la méthode classique / symbolique 
    def update_Q_values(self, current_s, action, reward, next_s, done): 
        self.Q_table[current_s, action] += self.lr * (reward + self.gamma * (np.max(self.Q_table[next_s]) - self.Q_table[current_s, action]))
        return 

    def update_Q_values_symbolic(self, current_s, action, reward, next_s, edition_distance, done):
        # On update la table pour tous les états possibles 
        for state in self.states: 
            ed_factor = np.exp(-(edition_distance[state, current_s])/self.p)
            self.Q_table[state, action] += self.lr * ed_factor * (reward + self.gamma * (np.max(self.Q_table[next_s] - self.Q_table[state, action])))
        return

    # Code le ration entre exploration et exploitation 
    def epsilon(self, time, nb_episodes):
        # Exemple de paramètres pour Epsilon : 
        standardized_time=(time-self.esp_param[0]*nb_episodes)/(self.esp_param[1]*nb_episodes)
        cosh=np.cosh(np.exp(-standardized_time))
        epsilon=1-(1/cosh+(time*self.esp_param[2]/nb_episodes))
        return self.esp_param[3]*epsilon 

    def print_espilon_function(self) :
        times =list(range(0,100))
        epsilon_values=[self.epsilon(time, self.esp_param, 100) for time in times]
        plt.plot(times, epsilon_values)
        plt.ylabel('Epsilon (Exploration %)')
        plt.xlabel('Time')
        plt.title('Epsilon function used')

    # Afficher de manière plus visuelle la Q_table 
    def print_heatmap_Q_table(self):
        plt.imshow(self.Q_table, cmap='hot', interpolation='nearest')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.show()
        
    def get_random_action(self):
        return random.choice(self.actions)


        





