## Stats:

- Health [0, 100]
- Energy [0, 100]
- Mood [0, 100] = Joy - Fear - Sadness - Anger
  - Joy [0, 100]
  - Fear [0, 50]
  - Sadness: [0, 50]
  - Anger: [0, 50]

 $r = f(h) + f(e) + f(m)$ with $f(x) = \sqrt{x}^{(2 - p)}$ with $p \in [0,2[$. When the value of a stat $s > 25$, $p = 0$ otherwise is a given parameter. 

We have this combinations of stats, that can be considered as an internal State $I$:

```
['Unhealthy', 'Bad Mood', 'Tired']
['Unhealthy', 'Bad Mood', 'Fit']
['Unhealthy', 'Good Mood', 'Tired']
['Unhealthy', 'Good Mood', 'Fit']
['Heatlhy', 'Bad Mood', 'Tired']
['Heatlhy', 'Bad Mood', 'Fit']
['Heatlhy', 'Good Mood', 'Tired']
['Heatlhy', 'Good Mood', 'Fit']]
```

The size of Q-Table will be : $|I| \times |S| \times |A|$. We now have the following updating rule of the Q-Table:
$$
Q[i, s, a_t] += \alpha e^{-d((i, s), (i_t, s_t))} (r_{t+1} + \gamma \text{max}_aQ(i_{t+1}, s_{t+1}, a) - Q[i, s, a_t])
$$
with
$$
d((i_1, s_1), (i_2, s_2)) = \frac{|i_1 \cap i_2|}{|I|}  + d(s_1, s_2)
$$
