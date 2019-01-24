# Reddit Project: Web APIs & Categorical Classification



### Problem Statement

The primary tasks of this Reddit project were 2-fold: 1. On the machine learning side, to extract and analyze the full range of posts from two subreddits and perform accurate binary classification between them 2. On the technological side, to test web scraping functionality, natural language processing tools and see what NLP tasks are possible and more so realistic for a range of data science models in a class project scope.

### Executive Summary



### Table of Contents

1. [Problem Statement](#problem-statement)
2. [Executive Summary](#executive-summary)
3. [Data Dictionaries](#data-dictionaries)
4. [Data Cleaning](#data-cleaning)
5. [Exploratory Data Analysis - Priority Variables](#exploratory-data-analysis-priority variables)
6. [Main Data Visualization](#main-data-visualization)
7. [Exploratory Data Analysis - Secondary Variables](#exploratory-data-analysis-secondary-variables)
8. [Secondary Visualizations](#secondary-visualizations)
9. [Conclusions and Recommendations](#conclusions-and-recommendations)
10. [Bibliography](#bibliography)







Assumptions:

1. Posts must be in English and include at least one word found in the English dictionary OR a word with higher than 0.01% term frequency in the overall posts corpus
2. The writer of the post as well as reddit moderators must think that the post belongs in the subreddit, as well as easily selected obvious posts that don't belong in cryptocurrency and environment subreddits. 
3. The most common cause that a post wouldn't belong to the subreddits is easily identifiable spam.
4. This also means excluding posts [removed] by reddit moderators  [deleted] by the user who posted it, be it for violating reddit terms of use or the specific subreddit terms of use.
5. The posts must have 4+ words from the English dictionary (at least with 0.01% corpus frequency), signifying some minimum of information represented in non-images (this was checked in a range of representative 0-3 Lemma count posts)



### Data Dictionary

400,000+ posts for Cryptocurrency subreddit from present to 1.5 years ago. All cryptocurrency data csv files have 'crypto' in filename.

~200,000 posts for Cryptocurrency subreddit from present to 8 years ago. All environment data csv files have 'env' in filename.



| Two Letter Words in English Dictionary                       |
| ------------------------------------------------------------ |
| AA, AB, AD, AE, AG, AH, AI, AL, AM, AN, AR, AS, AT, AW, AX, AY |
| BA, BE, BI, BO, BY, CH, DA, DE, DI, DO                       |
| EA, ED, EE, EF, EH, EL, EM, EN, ER, ES, ET, EW, EX           |
| FA, FE, FY, GI, GO, GU, HA, HE, HI, HM, HO                   |
| ID, IF, IN, IO, IS, IT, JA, JO                               |
| KA, KI, KO, KY, LA, LI, LO                                   |
| MA, ME, MI, MM, MO, MU, MY, NA, NE, NO, NU, NY               |
| OB, OD, OE, OF, OH, OI,OK, OM, ON, OO, OP, OR, OS, OU, OW, OX, OY |
| PA, PE, PI, PO, QI                                           |
| RE, SH, SI, SO, ST, TA, TE, TI, TO                           |
| UG, UH, UM, UN, UP, UR, US, UT                               |
| WE, WO, XI, XU, YA, YE, YO, YU, ZA                           |







```
Normalized Confusion Matrix: Naive Bayes for Environment Date Classification

            Predict 2010-12 | Predict 2013-15 | Predict 2016-18
Actual 2010-12    0.201     | 	0.0704		  | 	0.0453
Actual 2013-15    0.0809    |   0.185		  | 	0.0772
Actual 2016-18    0.0487    |   0.0831		  | 	0.208
```



```
Normalized Confusion Matrix: Naive Bayes for Crypto Date Classification

            Predict 9/5/17-12/17 | Predict 12/19/17-2/8/18 | Predict 7/1/18-10/1
Actual 9/5/17-12/17       0.149  |           0.116         | 		0.0560
Actual 12/19/17-2/8/18    0.0728 |           0.286         | 		0.0588
Actual 7/1/18-10/1        0.0342 |           0.0574        | 		0.169
```



```
Confusion Matrix: Best Naive Bayes Model on Test1 dataset

                 Predicted Crypto | Predicted Environment
Crypto Actual      76780 (61.6%)  |     695 (0.59%)
Environment Actual  1188 (0.95%)  |   45875 (36.8%)
```



```
Confusion Matrix: Best Gridsearch Random Forest on Test1 dataset

                 Predicted Crypto | Predicted Environment
     Crypto Actual  72668 (66.0%) |      794 (0.7%)
Environment Actual   1995 (1.8%)  |    34579 (31.4%)
```



### Conclusions and Next Steps

The project's NLP procedure has a high degree of accuracy for the task of binary classification of unrelated subreddits. The core procedure steps are lemmatizing all text including url and title, count vectorizing whereupon one can use a variety of machine learning models to train and predict based on the text corpus. Straightforward next steps include gradually expanding the number of subreddits from 2 to 3-10. The total count is of secondary concern compared to the amount of overlap between any two subreddit categories. It would be interesting to test on subreddits with substantial overlap but for subreddits that it is hard for humans to identify between the models would require more complex natural language processing features probably beyond n-grams and not within the scope of this project. Already, the computation power and time necessary for processing the large count vectorized matrices is substantial and the current training word corpus was cut in half due to my current hardware limitations. Another very intriguing next step would be to extend the spam classification approach to more subreddit categories especially to see how much similarity between types of spam posts there are between categories that are related to each other by different amounts.

It seems realistic that this project's core functionality if fully built-out could act as an automatic bot classifier that takes posts from a forum or other discussion website and implements appropriate tags and thread organization and which website users can then use to search and browse for related content. As data science improves and natural language processing techniques advance hopefully these sorts of approaches could be even extended to help create some of the World Wide Web Consortium's Semantic Web.



| 236956 | cryptoeconomics climate change transparent carbon market blockchain zengineeringpodcast christophe jospe nori climate change reverse | Crypto |
| ------ | ------------------------------------------------------------ | ------ |
| 223253 | palm beach research group file brokercheck finra org individual individual pdf | Crypto |
| 324305 | here pollution shipping container freight truck cause network medium network here pollution shipping container freight truck cause | Crypto |
| 313634 | elon musk want use rocket intercity travel earth themerkle elon musk want use rocket city city travel earth | Crypto |

```
(2923, climate)            45793
(10835, news)              31200
(2658, change)             29381
(11379, org)               27000
(5628, energy)             18030
(10818, new)               16323
(874, article)             15695
(17569, world)             15144
(17278, water)             14099
(11229, oil)               12740
(5714, environment)        12516
(17732, year)              11963
(7103, global)             11553
(13881, say)               10847
(13936, science)           10151
(7539, have)               10113
(5715, environmental)       9823
(17240, warm)               9191
(12297, power)              8777
(3005, coal)                8353
(1839, blog)                8268
(16316, trump)              8057
(7284, green)               8019
(2431, carbon)              7876
(14663, solar)              7861
(16805, use)                7815
(15151, study)              7790
(12191, pollution)          7520
(15072, story)              7426
(5742, epa)                 7215
(13955, scientist)          7215
(14971, state)              7130
(5280, earth)               7106
(6923, gas)                 6950
(12065, plant)              6878
(9768, make)                6645
(15950, time)               6570
(7632, help)                6301
(11824, people)             6285
(15771, theguardian)        6130
(14008, sea)                6116
(6788, fuel)                6054
(17267, waste)              6031
(1528, big)                 5968
(12069, plastic)            5901
(13870, save)               5870
(361, air)                  5811
(13321, report)             5716
(8924, just)                5694
(5559, emission)            5654
(11173, ocean)              5648
(12055, plan)               5452
(6670, fracking)            5427
(3298, company)             5216
(6570, food)                5191
(2893, clean)               5105
(2350, california)          4948
(11065, nuclear)            4883
(17296, way)                4825
(10749, need)               4709
(2848, city)                4688
(4301, day)                 4653
(2761, china)               4640
(8314, industry)            4604
(14826, spill)              4548
(12025, pipeline)           4545
(2264, business)            4401
(822, arctic)               4344
(14726, source)             4325
(17272, watch)              4294
(9356, level)               4274
(2527, cause)               4266
(7909, human)               4225
(10671, national)           4128
(9017, kill)                4111
(13557, rise)               4105
(11123, obama)              4104
(13052, record)             4073
(7994, ice)                 4070
(14787, specie)             4045
(17231, want)               4035
(1232, ban)                 4031
(9068, know)                3992
(6606, forest)              3989
(7212, government)          3921
(17472, wind)               3877
(1185, bad)                 3876
(7275, great)               3868
(12544, project)            3849
(7576, health)              3846
(11112, nytimes)            3833
(17773, youtube)            3827
(12057, planet)             3827
(16225, tree)               3806
(6216, farm)                3776
(9197, large)               3703
(9393, life)                3667
(6646, fossil)              3655
(6846, future)              3633
(7680, high)                3571
```

```
(3877, crypto)                       102277
(1610, bitcoin)                       78235
(3048, coin)                          64983
(13275, remove)                       57119
(1789, blockchain)                    49936
(3922, cryptocurrency)                48243
(5996, exchange)                      40761
(10818, new)                          39404
(9871, market)                        38459
(10835, news)                         36642
(7539, have)                          35953
(16805, use)                          34141
(8924, just)                          33332
(16018, token)                        31654
(12432, price)                        27767
(10035, medium)                       26516
(2146, btc)                           26481
(15950, time)                         24456
(13082, redd)                         24327
(8006, ico)                           24006
(11824, people)                       23966
(9768, make)                          22749
(9068, know)                          22254
(17210, wallet)                       22072
(2286, buy)                           21431
(12544, project)                      21239
(9589, look)                          20737
(15863, think)                        20334
(10395, money)                        20137
(12073, platform)                     19425
(16154, trading)                      19132
(17231, want)                         19071
(5866, ethereum)                      18812
(4121, currency)                      18757
(8605, invest)                        18489
(13881, say)                          17353
(10749, need)                         17206
(8884, jpg)                           17108
(14088, sell)                         17060
(4301, day)                           16518
(17773, youtube)                      16103
(17272, watch)                        16002
(12266, post)                         15854
(7024, get)                           15783
(14961, start)                        15463
(7175, good)                          15429
(17732, year)                         15315
(17558, work)                         15139
(5850, eth)                           15012
(1489, best)                          14979
(5064, don)                           14926
(16398, twitter)                      14890
(17569, world)                        14841
(17296, way)                          14637
(10805, network)                      14578
(9228, launch)                        14317
(3052, coinbase)                      14085
(6846, future)                        14047
(1528, big)                           13934
(3921, cryptocurrencies)              13845
(3247, come)                          13645
(7632, help)                          13626
(1548, binance)                       13338
(8182, important)                     13286
(1252, bank)                          13274
(16175, transaction)                  13101
(16813, user)                         12628
(16888, value)                        11950
(15565, team)                         11651
(5933, event)                         11607
(8091, ift)                           11531
(12794, question)                     11507
(16144, trade)                        11500
(3298, company)                       11468
(3291, community)                     11225
(15588, technology)                   11223
(13533, right)                        11065
(93, account)                         11019
(14726, source)                       10802
(10248, mining)                       10747
(2487, cash)                          10727
(1296, base)                          10678
(16332, try)                          10488
(13552, ripple)                       10485
(8616, investment)                    10459
(15861, thing)                        10330
(7399, guy)                           10328
(15312, support)                      10210
(8618, investor)                      10170
(9464, list)                          10123
(10425, month)                         9987
(7680, high)                           9957
(14986, status)                        9923
(17343, week)                          9869
(12978, really)                        9715
(16738, update)                        9644
(16003, today)                         9637
(3760, create)                         9493
(15885, thought)                       9481
(737, app)                             9327
```

```
14      9107 most frequent word counts
13      8920
15      8769
12      8616
16      8112
11      7857
17      7549
10      7097
18      6456
9       6075
19      5787
8       5098
20      4883
21      4349
7       3922
22      3730
23      3325
6       3196
24      2866
25      2702
26      2481
27      2357
28      2175
5       2162
29      1907
30      1837
4       1637
31      1630
32      1444
33      1212
34      1024
3        968
35       889
36       710
2        579
37       539
38       459
39       345
40       268
1        227
42       172
41       172
43       147
0        132
44        99
```

```
362444    1349 (env posts with highest word counts)
387356    1328
106879    1172
185309    1134
2898      1092
384674    1082
350164    1025
185173     857
341982     800
400639     777
385376     766
4989       755
164918     755
298304     749
437278     740
322469     725
40648      724
28176      721
311844     716
23654      709
334687     698
420123     687
114138     670
384426     667
15366      644
264506     635
174646     626
292799     621
198703     599
160241     589
81380      580
155783     570
51227      561
425226     561
45741      556
31954      555
28177      526
239961     523
189451     522
274436     522
133641     519
51052      517
136006     514
269817     512
293920     511
336706     509
189525     502
173356     499
49530      498
252478     491
162780     484
118384     483
412610     482
44778      479
98896      477
242191     476
156626     475
159270     473
292787     470
338631     468
373663     467
263200     466
289165     465
350830     463
20591      462
302869     460
110667     454
114547     454
231423     453
195824     450
99369      449
372536     444
74139      441
96984      440
328983     437
61468      434
121021     433
205277     429
407724     427
259504     424
286139     423
408730     422
314133     417
108417     405
437197     404
280920     403
230912     399
397621     399
245574     398
34519      393
182162     392
290834     390
175637     390
172997     387
170390     386
271857     384
37278      383
20909      380
59969      377
62389      373
```



https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

Vik Paruchuri, August 5, 2017. http://www.pybloggers.com/2017/08/using-pandas-with-large-data/

https://blog.scrapinghub.com/2014/03/26/optimizing-memory-usage-of-scikit-learn-models-using-succinct-tries

---

### Why we choose this project for you?
This project covers three of the biggest concepts we cover in the class: Classification Modeling, Natural Language Processing and Data Wrangling/Acquisition.

Part 1 of the project focuses on **Data wrangling/gathering/acquisition**. This is a very important skill as not all the data you will need will be in clean CSVs or a single table in SQL.  There is a good chance that wherever you land you will have to gather some data from some unstructured/semi-structured sources; when possible, requesting information from an API, but often scraping it because they don't have an API (or it's terribly documented).

Part 2 of the project focuses on **Natural Language Processing** and converting standard text data (like Titles and Comments) into a format that allows us to analyze it and use it in modeling.

Part 3 of the project focuses on **Classification Modeling**.  Given that project 2 was a regression focused problem, we needed to give you a classification focused problem to practice the various models, means of assessment and preprocessing associated with classification.   
