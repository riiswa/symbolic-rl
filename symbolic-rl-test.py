from recordtype import recordtype
from random import random, choice
from pprint import pprint

hunger_rate = 10
lives = 5

Emotion = recordtype("Emotion", "value")

Pain = Emotion(-2)

Pleasure = Emotion(2)

emotions = [Pain, Pleasure]

Action = recordtype("Action", "link_name")

Eat = Action("is_eaten")

DoNothing = Action("do_nothing")

actions = [Eat, DoNothing]

Event = recordtype("Event", "name")

Cactus = Event("cactus")

Apple = Event("apple")

events = [Cactus, Apple]


Link = recordtype("Link", "frome to link_name weight")

links = []

def increase_link_weight(_from, to, link_name):
	filtered_link = list(filter(lambda link: link.frome == _from and link.to == to, links))
	if filtered_link:
		filtered_link[0].weight += 1
	else:
		links.append(Link(_from, to, link_name, 1))


def interact_with_env(event, action):
	global hunger_rate, lives
	if event == Cactus:
		if action == Eat:
			hunger_rate = max(hunger_rate, (hunger_rate + 1) % 10)
		
			if random() >= 0.05: # 5% chance of avoiding being stung
				increase_link_weight(Eat, Pain, "gives")
				increase_link_weight(Cactus, Pain, action.link_name)
				lives -= 1
			else:
				increase_link_weight(Eat, Pleasure, "gives")
				increase_link_weight(Cactus, Pleasure, action.link_name)
		elif action == DoNothing:
			increase_link_weight(Cactus, Pleasure, action.link_name)
			increase_link_weight(DoNothing, Pleasure, "gives")
	elif event == Apple:
		if action == Eat:
			if random() >= 0.05: # 5% chance of finding a rotten apple 
				hunger_rate = max(hunger_rate, (hunger_rate + 1) % 10)
				increase_link_weight(Eat, Pleasure, "gives")
				increase_link_weight(Apple, Pleasure, action.link_name)


def explore(N):
	global hunger_rate, lives
	i = 0

	while i < N and hunger_rate >= 0 and lives >= 0:
		event = choice(events)
		action = choice(actions)
		interact_with_env(event, action)
		if i % 2 == 0:
			hunger_rate -= 1
		i += 1

explore(100)

for link in links:
	print(link.frome, "->", link.to, link.weight)

			
		


