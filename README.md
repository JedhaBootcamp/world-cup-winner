# World Cup Winner
Voici un algorithme de Deep Learning qui permet de prédire qui va gagner la coupe du monde

# Introduction

Du moment qu'il y a des données, les Data Sciences peuvent être utilisées. Si certains d'entre vous ont l'âme d'un parieur et souhaitent éclairer leurs décisions par des statistiques, nous avons créé un algorithme qui permet de prédire l'équipe gagnante d'un match de la coupe du monde.

Cet algorithme vient, en partie de [Dr James Bond](https://www.kaggle.com/agostontorok) et de son projet sur Kaggle : [Soccer World Cup 2018 Winner](https://www.kaggle.com/agostontorok/soccer-world-cup-2018-winner/notebookhttps://www.kaggle.com/agostontorok/soccer-world-cup-2018-winner/notebook). Celui-ci se basait cependant exclusivement sur les classements Fifa de chaque équipe.

Nous y avons ajouté les statistiques moyennes de chaque joueur et de chaque équipe dans l'équation. En plus de cela, nous avons opté pour un modèle de Deep Learning de Réseau de Neurones pour gagner en précision.

Voici donc une explication du projet, étape par étape. Pour ceux qui souhaitent simplement avoir les résultats prévisionnels, vous pouvez scroller directement à la fin de cet article ;)

# Collection des Données

En ce qui concerne les données que nous avons récoltées, celles-ci viennent principalement de bases Kaggle :

1. Nous avons les rangs FIFA de 1993 à 2018 donnés par [Tadhg Fitzgerald](https://www.kaggle.com/tadhgfitzgerald)
2. L'historique des matches de football depuis 1872 donné par [Mart Jürisoo](https://www.kaggle.com/martj42)
3. Les statistiques de chaque équipe depuis 2018 tirées de [Wikipédia](https://en.wikipedia.org/wiki/All-time_table_of_the_FIFA_World_Cup)
4. Les statistiques des joueurs tirées de [Kaggle](https://www.kaggle.com/antoinekrajnc/soccer-players-statistics)
5. Les futurs matches de la coupe de 2018 donnés par [Nuggs](https://www.kaggle.com/ahmedelnaggar)



C'est avec ceci que nous allons entamé notre analyse. Importons donc les différents Datasets :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

matches = pd.read_csv("results.csv")
rankings = pd.read_csv("fifa_ranking.csv")
world_cup_matches = pd.read_csv("World Cup 2018 Dataset.csv")
players = pd.read_csv("FullData.csv")
all_time_stats = pd.read_csv("all_time_fifa_statistics.csv")
```


Nous n'avons pas besoin de toutes les données dans chaque fichier. Certains noms de pays diffèrent en fonction des années (l'Allemagne comptait comme deux pays avant la chute du mur de Berlin en 1989). Nous allons donc commencer une première phase de nettoyage

```python
rankings = rankings.loc[:,['rank',
                           'country_full',
                           'country_abrv',
                           'cur_year_avg_weighted',
                           'rank_date',
                           'two_year_ago_weighted',
                           'three_year_ago_weighted']]
rankings = rankings.replace({"IR Iran": "Iran"})
rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])

matches = matches.replace({"Germany DR": "Germany", "China": "China PR"})
matches["date"] = pd.to_datetime(matches["date"])

world_cup_matches = world_cup_matches.loc[:, ['Team',
                                              'Group',
                                              'First match \nagainst',
                                              'Second match\n against',
                                              'Third match\n against']]
world_cup_matches = world_cup_matches.dropna(how='all')
world_cup_matches = world_cup_matches.replace({"IRAN": "Iran",
                               "Costarica": "Costa Rica",
                               "Porugal": "Portugal",
                               "Columbia": "Colombia",
                               "Korea" : "Korea Republic"})
world_cup_matches = world_cup_matches.set_index('Team')
```

Vu la quantité de données que l'on possède ainsi que le peu de données manquantes, nous décidons de simplement effacer les lignes où il y a des données manquantes. Finissons d'importer les statistiques des joueurs

```python
players = players.loc[:, ["Nationality",
                            "Rating",
                            "Age",
                            "Weak_foot",
                            "Skill_Moves",
                            "Ball_Control",
                            "Dribbling",
                            "Marking",
                            "Sliding_Tackle",
                            "Standing_Tackle",
                            "Aggression",
                            "Reactions",
                            "Attacking_Position",
                            "Interceptions",
                            "Vision",
                            "Composure",
                            "Crossing",
                             "Short_Pass",
                             "Long_Pass",
                             "Acceleration",
                             "Speed",
                             "Stamina",
                             "Strength",
                             "Balance",
                             "Agility",
                             "Jumping",
                             "Heading",
                             "Shot_Power",
                             "Finishing",
                             "Long_Shots",
                             "Curve",
                             "Freekick_Accuracy",
                             "Penalties",
                             "Volleys"]]
players = players.dropna(how="all")
grouped = players.groupby(["Nationality"], as_index = False)
players = grouped.aggregate(np.mean)
```

La fin de la partie du code sert à calculer la moyenne des statistiques des joueurs dans chaque équipe pour que l'on puisse ensuite les intégrer dans le comparatif entre chaque pays.

# Préparation des données

Nos données sont maintenant importées mais nous devrons les fusionner pour que notre algorithme puisse apprendre des différentes statistiques. Il faudra le faire en plusieurs étapes.

D'abord, les rangs et les dates des matches ne correspondent pas exactement. En effet, nous avons les rangs au mois-le-mois alors que nous avons une date au jours près pour les matches. Il faudra donc créer un classement au jour-le-jour pour que l'on puisse fusionner nos colonnes.

Une fois que ceci est fait, nous faisons un premier ```merge``` (fusion).

```python

rankings = rankings.set_index(['rank_date'])\
            .groupby(['country_full'], group_keys=False)\
            .resample('D').first()\
            .fillna(method='ffill')\
            .reset_index()


matches = matches.merge(rankings,
                        left_on=['date', 'home_team'],
                        right_on=['rank_date', 'country_full'])
matches = matches.merge(rankings,
                        left_on=['date', 'away_team'],
                        right_on=['rank_date', 'country_full'],
                        suffixes=('_home', '_away'))

```

En ce qui concerne les statistiques des joueurs et des équipes au général, nous n'avons pas besoin de toucher aux dates. On passera donc directement à l'étape merge

```python
matches = matches.merge(players,
                       left_on =["home_team"],
                       right_on = ["Nationality"])

matches = matches.merge(players,
                        left_on = ['away_team'],
                        right_on = ["Nationality"],
                        suffixes = ('_home', "_away"))

matches = matches.merge(all_time_stats,
                       left_on = ["home_team"],
                       right_on = ["Country"])

matches = matches.merge(all_time_stats,
                       left_on = ["away_team"],
                        right_on = ["Country"],
                       suffixes = ("_home", "_away"))
```

Comment allons nous evaluer les différentes équipes qui s'affrontent ? Un moyen simple est de prendre la différence de chaque statistiques entre les équipes. Par exemple, nous allons prendre la différence de position dans les classements FIFA, la différence d'âge entre les joueurs etc. Ce processus est un peu fastidieux car il faudra tout faire à la main mais le voici :

```python
matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches["point_difference"] = matches['weighted_points_home'] - matches['weighted_points_away']
matches["rating_difference"] = matches["Rating_home"] - matches["Rating_away"]
matches["Age_difference"] = matches["Age_home"] - matches["Age_away"]
matches["Weak_foot_difference"] = matches["Weak_foot_home"] - matches["Weak_foot_away"]
matches["Skill_Moves_difference"] = matches["Skill_Moves_home"] - matches["Skill_Moves_away"]
matches["Ball_Control_difference"] = matches["Ball_Control_home"] - matches["Ball_Control_away"]
matches["Dribbling_difference"] = matches["Dribbling_home"] - matches["Dribbling_away"]
matches["Marking_difference"] = matches["Marking_home"] - matches["Marking_away"]
matches["Sliding_Tackle_difference"] = matches["Sliding_Tackle_home"] - matches["Sliding_Tackle_away"]
matches["Standing_Tackle_difference"] = matches["Standing_Tackle_home"] - matches["Standing_Tackle_away"]
matches["Aggression_difference"] = matches["Aggression_home"] - matches["Aggression_away"]
matches["Reactions_difference"] = matches["Reactions_home"] - matches["Reactions_away"]
matches["Attacking_Position_difference"] = matches["Attacking_Position_home"] - matches["Attacking_Position_away"]
matches["Interceptions_difference"] = matches["Interceptions_home"] - matches["Interceptions_away"]
matches["Vision_difference"] = matches["Vision_home"] - matches["Vision_away"]
matches["Composure_difference"] = matches["Composure_home"] - matches["Composure_away"]
matches["Crossing_difference"] = matches["Crossing_home"] - matches["Crossing_away"]
matches["Short_Pass_difference"] = matches["Short_Pass_home"] - matches["Short_Pass_away"]
matches["Long_Pass_difference"] = matches["Long_Pass_home"] - matches["Long_Pass_away"]
matches["Stamina_difference"] = matches["Stamina_home"] - matches["Stamina_away"]
matches["Penalties_difference"] = matches["Penalties_home"] - matches["Penalties_away"]
matches["Acceleration_difference"] = matches["Acceleration_home"] - matches["Acceleration_away"]
matches["Speed_difference"] = matches["Speed_home"] - matches["Speed_away"]
matches["Strength_difference"] = matches["Strength_home"] - matches["Strength_away"]
matches["Balance_difference"] = matches["Balance_home"] - matches["Balance_away"]
matches["Agility_difference"] = matches["Agility_home"] - matches["Agility_away"]
matches["Jumping_difference"] = matches["Jumping_home"] - matches["Jumping_away"]
matches["Heading_difference"] = matches["Heading_home"] - matches["Heading_away"]
matches["Shot_Power_difference"] = matches["Shot_Power_home"] - matches["Shot_Power_away"]
matches["Finishing_difference"] = matches["Finishing_home"] - matches["Finishing_away"]
matches["Long_Shots_difference"] = matches["Long_Shots_home"] - matches["Long_Shots_away"]
matches["Curve_difference"] = matches["Curve_home"] - matches["Curve_away"]
matches["Freekick_Accuracy_difference"] = matches["Freekick_Accuracy_home"] - matches["Freekick_Accuracy_away"]
matches["Volleys_difference"] = matches["Volleys_home"] - matches["Volleys_away"]
matches["Part's_difference"] = matches["Part's_home"] - matches["Part's_away"]
matches["Played_difference"] = matches["Played_home"] - matches["Played_away"]
matches["Won_difference"] = matches["Won_home"] - matches["Won_away"]
matches["Drawn_difference"] = matches["Drawn_home"] - matches["Drawn_away"]
matches["Lost_difference"] = matches["Lost_home"] - matches["Lost_away"]
matches["Goal_Difference_difference"] = matches["Goal Difference_home"] - matches["Goal Difference_away"]
matches["Points_difference"] = matches["Points_home"] - matches["Points_away"]
matches["Average_points_difference"] = matches["Average_points_home"] - matches["Average_points_away"]
matches['is_won'] = matches['score_difference'] > 0 # take draw as lost
matches['is_stake'] = matches['tournament'] != 'Friendly'
```

La gestion de chacune de nos variables qui va s'en suivre va de même être quelque peu longue et il existe très certainement des moyens de gérer cela d'une meilleure façon mais, par contrainte de temps, nous avons préféré procéder ainsi.

# Construction du modèle

Commencons par séparer nos variables dépendantes de notre variable dépendante (ce que l'on veut prédire.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

X = matches.loc[:,['average_rank',
                    'rank_difference',
                    "point_difference",
                    'is_stake',
                    "rating_difference",
                     "Age_difference",
                    "Weak_foot_difference",
                     "Skill_Moves_difference",
                    "Ball_Control_difference",
                     "Dribbling_difference",
                     "Marking_difference",
                     "Sliding_Tackle_difference",
                     "Standing_Tackle_difference",
                     "Aggression_difference",
                     "Reactions_difference",
                     "Interceptions_difference",
                     "Vision_difference",
                   "Crossing_difference",
                     "Short_Pass_difference",
                     "Long_Pass_difference",
                    "Stamina_difference",
                     "Penalties_difference",
                     "Acceleration_difference",                   
                     "Speed_difference",
                    "Strength_difference",
                    "Balance_difference",
                     "Agility_difference",
                     "Jumping_difference",
                    "Heading_difference",
                     "Shot_Power_difference",
                    "Finishing_difference",
                   "Long_Shots_difference",
                     "Curve_difference",
                    "Freekick_Accuracy_difference",
                     "Volleys_difference",
                     "Won_difference",
                     "Drawn_difference",
                     "Lost_difference",
                     "Average_points_difference",
                  ]]
y = matches['is_won']
```

Notre variable dépendante y est pour l'instant de type boolean mais pour que notre modèle puisse apprendre, il nous faut des chiffres. On utilisera donc la fonction get_dummies

```python
y = pd.get_dummies(y, drop_first = True)
y.head()
```

Nous pouvons maintenant séparer notre dataset en un training set et un test set. Rien de bien compliqué, nous utiliserons simplement scikitlearn.

```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```

Nous pouvons maintenant entrainer notre modèle de Machine Learning. Alors que Dr James Bond utilise une régression logistique (qui fonctionne très bien), nous avons tenté d'utiliser un modèle de Réseau de Neurones qui s'avère être un poil plus précis. C'est donc pour cela que nous avons opté pour ceci.

```python
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier_ann = Sequential()

# Adding the input layer and the first hidden layer
classifier_ann.add(Dense(activation="relu", input_dim=39, units=5, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier_ann.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier_ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier_ann.fit(X_train, y_train, batch_size = 10, epochs = 10)
```
Nous n'avons pas créé trop d'epochs pour ne pas créer un phénomène d'overfitting (modèle trop entraîné sur le training set). Nous obtenons pour l'instant un score de précision aux alentours de 65%. Ce qui n'est pas trop mal étant donné que le niveau des équipes est difficile à déterminer.

Nous donnerons à la fin de l'article les probabilités exactes, celles-ci sont plutôt proches des 50% c'est pour cela que la modèle a aussi du mal à prédire avec plus de précision qui va gagner chaque match.

# Application du modèle

Maintenant que notre modèle est prêt, utilisons le sur les équipes qui vont jouer dans cette coupe du monde.

Pour ce faire, nous devrons reprendre les étapes de notre partie préparation de données et les intégrer dans notre Dataset.

```python
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) &
                                    rankings['country_full'].isin(world_cup_matches.index.unique())]


world_cup_rankings = world_cup_rankings.merge(players,
                                              left_on =["country_full"],
                                              right_on = ["Nationality"])


world_cup_rankings = world_cup_rankings.merge(all_time_stats,
                       left_on = ["country_full"],
                       right_on = ["Country"])


world_cup_rankings = world_cup_rankings.set_index(['country_full'])
world_cup_rankings
```

Nous devrons ensuite intégrer les différentes combinaisons possibles de matches en considérant les poules qui ont été faites.

```python
X_test = pd.DataFrame(X_test)

from itertools import combinations

world_cup_matches['points'] = 0
world_cup_matches['total_prob'] = 0

for group in set(world_cup_matches['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup_matches.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        row = pd.DataFrame(np.array([[np.nan,
                                      np.nan,
                                      np.nan,
                                      True,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                      np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan,
                                     np.nan]]), columns= X_test.columns)


        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        home_rating = world_cup_rankings.loc[home,'Rating']
        home_age = world_cup_rankings.loc[home,'Age']
        home_weak_foot = world_cup_rankings.loc[home,'Weak_foot']
        home_Skill_Moves = world_cup_rankings.loc[home,'Skill_Moves']
        home_Ball_Control = world_cup_rankings.loc[home,'Ball_Control']
        home_Dribbling = world_cup_rankings.loc[home, 'Dribbling']
        home_Marking = world_cup_rankings.loc[home, 'Marking']
        home_Sliding_Tackle = world_cup_rankings.loc[home, 'Sliding_Tackle']
        home_Standing_Tackle = world_cup_rankings.loc[home, 'Standing_Tackle']
        home_Aggression = world_cup_rankings.loc[home, 'Aggression']
        home_Reactions = world_cup_rankings.loc[home, 'Reactions']
        home_Interceptions = world_cup_rankings.loc[home, 'Interceptions']
        home_Vision = world_cup_rankings.loc[home, "Vision"]
        home_Crossing = world_cup_rankings.loc[home, "Crossing"]
        home_Short_Pass = world_cup_rankings.loc[home, "Short_Pass"]
        home_Long_Pass = world_cup_rankings.loc[home, "Long_Pass"]                 
        home_stamina = world_cup_rankings.loc[home,'Stamina']
        home_Penalties = world_cup_rankings.loc[home,'Penalties']
        home_Acceleration = world_cup_rankings.loc[home,'Acceleration']        
        home_speed = world_cup_rankings.loc[home,'Speed']
        home_strength = world_cup_rankings.loc[home,'Strength']
        home_balance = world_cup_rankings.loc[home,'Balance']
        home_agility = world_cup_rankings.loc[home,'Agility']
        home_jumping = world_cup_rankings.loc[home,'Jumping']
        home_won = world_cup_rankings.loc[home,'Won']
        home_drawn = world_cup_rankings.loc[home,'Drawn']
        home_lost = world_cup_rankings.loc[home,'Lost']
        home_average_points = world_cup_rankings.loc[home,'Average_points']
        home_Heading = world_cup_rankings.loc[home,'Heading']
        home_Shot_Power = world_cup_rankings.loc[home,'Shot_Power']
        home_Finishing = world_cup_rankings.loc[home,'Finishing']
        home_Long_Shots = world_cup_rankings.loc[home,'Long_Shots']
        home_Curve = world_cup_rankings.loc[home,'Curve']
        home_Freekick_Accuracy = world_cup_rankings.loc[home,'Freekick_Accuracy']
        home_Volleys = world_cup_rankings.loc[home,'Volleys']        

        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        opp_rating = world_cup_rankings.loc[away,'Rating']
        opp_age = world_cup_rankings.loc[away,'Age']
        opp_weak_foot = world_cup_rankings.loc[away,'Weak_foot']
        opp_Skill_Moves = world_cup_rankings.loc[away,'Skill_Moves']
        opp_Ball_Control = world_cup_rankings.loc[away,'Ball_Control']      
        opp_Dribbling = world_cup_rankings.loc[away, 'Dribbling']
        opp_Marking = world_cup_rankings.loc[away, 'Marking']
        opp_Sliding_Tackle = world_cup_rankings.loc[away, 'Sliding_Tackle']
        opp_Standing_Tackle = world_cup_rankings.loc[away, 'Standing_Tackle']
        opp_Aggression = world_cup_rankings.loc[away, 'Aggression']
        opp_Reactions = world_cup_rankings.loc[away, 'Reactions']
        opp_Interceptions = world_cup_rankings.loc[away, 'Interceptions']
        opp_Vision = world_cup_rankings.loc[away, "Vision"]
        opp_Crossing = world_cup_rankings.loc[away, "Crossing"]
        opp_Short_Pass = world_cup_rankings.loc[away, "Short_Pass"]
        opp_Long_Pass = world_cup_rankings.loc[away, "Long_Pass"]                 
        opp_stamina = world_cup_rankings.loc[away,'Stamina']
        opp_Penalties = world_cup_rankings.loc[away,'Penalties']
        opp_Acceleration = world_cup_rankings.loc[away,'Acceleration']      
        opp_speed = world_cup_rankings.loc[away, 'Speed']
        opp_strength = world_cup_rankings.loc[away,'Strength']
        opp_balance = world_cup_rankings.loc[away,'Balance']
        opp_agility = world_cup_rankings.loc[away,'Agility']
        opp_jumping = world_cup_rankings.loc[away,'Jumping']
        opp_won = world_cup_rankings.loc[away,'Won']
        opp_drawn = world_cup_rankings.loc[away,'Drawn']
        opp_lost = world_cup_rankings.loc[away,'Lost']
        opp_average_points = world_cup_rankings.loc[away,'Average_points']
        opp_Heading = world_cup_rankings.loc[away,'Heading']
        opp_Shot_Power = world_cup_rankings.loc[away,'Shot_Power']
        opp_Finishing = world_cup_rankings.loc[away,'Finishing']
        opp_Long_Shots = world_cup_rankings.loc[away,'Long_Shots']
        opp_Curve = world_cup_rankings.loc[away,'Curve']
        opp_Freekick_Accuracy = world_cup_rankings.loc[away,'Freekick_Accuracy']
        opp_Volleys = world_cup_rankings.loc[away,'Volleys']

        row['average_rank'] = (home_rank + opp_rank) / 2
        row['rank_difference'] = home_rank - opp_rank
        row['point_difference'] = home_points - opp_points
        row['rating_difference'] = home_rating - opp_rating
        row['Age_difference'] = home_age - opp_age
        row['Weak_foot_difference'] = home_weak_foot - opp_weak_foot
        row['Skill_Moves_difference'] = home_Skill_Moves - opp_Skill_Moves
        row['Ball_Control_difference'] = home_rating - opp_rating        
        row["Dribbling_difference"] = home_Dribbling - opp_Dribbling
        row['Marking_difference'] = home_Marking - opp_Marking
        row['Sliding_Tackle_difference'] = home_Sliding_Tackle - opp_Sliding_Tackle
        row["Standing_Tackle_difference"] = home_Standing_Tackle - opp_Standing_Tackle
        row["Aggression_difference"] = home_Aggression - opp_Aggression
        row["Reactions_difference"] = home_Reactions - opp_Reactions
        row["Interceptions_difference"] = home_Interceptions - opp_Interceptions
        row["Vision_difference"] = home_Vision - opp_Vision
        row["Crossing_difference"] = home_Crossing - opp_Crossing
        row["Short_Pass_difference"] = home_Short_Pass - opp_Short_Pass
        row["Long_Pass_difference"] = home_Long_Pass - opp_Long_Pass
        row['Stamina_difference'] = home_stamina - opp_stamina
        row['Penalties_difference'] = home_Penalties - opp_Penalties
        row['Acceleration_difference'] = home_Acceleration - opp_Acceleration
        row['Speed_difference'] = home_speed - opp_speed
        row['Strength_difference'] = home_strength - opp_strength
        row['Balance_difference'] = home_balance - opp_balance
        row['Agility_difference'] = home_agility - opp_agility
        row['Jumping_difference'] = home_jumping - opp_jumping
        row['Won_difference'] = home_won - opp_won
        row['Drawn_difference'] = home_drawn - opp_drawn
        row['Lost_difference'] = home_lost - opp_lost
        row['Average_points_difference'] = home_average_points - opp_average_points
        row['Heading_difference'] = home_Heading - opp_Heading
        row['Shot_Power_difference'] = home_Shot_Power - opp_Shot_Power
        row['Finishing_difference'] = home_Finishing - opp_Finishing
        row['Long_Shots_difference'] = home_Long_Shots - opp_Long_Shots
        row['Curve_difference'] = home_Curve - opp_Curve
        row['Freekick_Accuracy_difference'] = home_Freekick_Accuracy - opp_Freekick_Accuracy
        row['Volleys_difference'] = home_Volleys - opp_Volleys



        home_win_prob = classifier_ann.predict_proba(row)[:,0][0]
        world_cup_matches.loc[home, 'total_prob'] += home_win_prob
        world_cup_matches.loc[away, 'total_prob'] += 1-home_win_prob

        points = 0
        if home_win_prob <= 0.5:
            print("{} wins with {:.2f}".format(away, 1-home_win_prob))
            world_cup_matches.loc[away, 'points'] += 3
        if home_win_prob > 0.5:
            points = 3
            world_cup_matches.loc[home, 'points'] += 3
            print("{} wins with {:.2f}".format(home, home_win_prob))
```

C'est une grosse loupe avec beaucoup de code assez long à entrer à la main mais voici nos résultats pour les matches de poules !

L'algorithme ne prédit pas encore les matches nuls donc il se peut qu'il y a des erreurs à cause de cela mais nous sommes pour l'instant à 60% de prédictions justes !


Si vous êtes intéressé à l'idée d'apprendre les Data Sciences, regardez notre site : [Jedha.co](https://jedha.co)
