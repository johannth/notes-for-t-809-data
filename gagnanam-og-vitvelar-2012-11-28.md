% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 28. November 2012

Upprifjun
===============================

Þekkja vel multivariant Gaussian distribution, kaflar 1.1 - 1.5

Úr kafla 1.5.
1. Reikna líkur á að eitthvað gerist
2. Hugsa um áhættuna sem felst í því að taka ákvörðunina

Muna um þrjár leiðir til að leysa ákvörðunartöku (bls 43)
1. Beint (sjá kafla 4.2)
2. Bayes (sjá kafla 4.3)
3. Discriminant fall

Maximum likelihood mat á stikum (t.d. með því að diffra log og svo frv.)
Sequential estimation, hvernig uppfærum við stikana ef við fáum einn aukapunkt


Leiða út formúluna fyrir w í 4 kafla með meðaltölum og samvikafylkjum

stikamöt eru grunnurinn fyrir supervised learning ??? hvernig vélin okkar lærir þegar út á hólminn er komið.

Kafli 3
----------
Aðhvarfsgreining og flokkun

Kafli 3.1.1 er áherslukafli
3.1.2 umræða um rúmfræðilega túlkun, hvernig y er nálgun á t
3.1.3 svipar til gradient descent og tauganet

3.1.4 Regularization, mjög mikilvægt, key concept, tengist overfitting og underfitting. Hvað gerist ef við erum með of lítið af gögnum og of mikið af stikum. Setur skorður á stuðlana svo þeir geta ekki vaxið um of. Erum alltaf að reyna að lágmarka villuna á gögn sem höfum ekki séð. Sjá jöfnu 3.24, þá setur regularization vigt á vogtölurnar, getum fjölgað vogtölum án þess að hafa áhrif á overfitting. Erfitt að setja upp regularization fyrir tauganet.

3.3 ekki á prófi nema kannski hver er munurinn á maximum likelihood stikamati og bayesísku stikamati.

maximum likelihood stikamat gerir ráð fyrir að stikarnir séu tölur sem þarf að finna en í bayesísku stikamati er gert ráð fyrir að stikarnir hafi drefingu sem þarf að finna, þ.e. stikarnir eru random variables.

Í þessu kúrsi lögð áhersla á maximum likelihood stikamat, önnur teoría með bayesískt stikamat

Kafli 4
---------
Class conditional propability

Gaussian
    1. $\mvSigma_i = \sigma^2 \I$ -> línulegur flokkari
    2. $\mvSigma_i = \mvSigma$ -> línulegur flokkari
    3. $\mvSigma_i$ -> quadratic flokkari

Vel fjallað um ofangreint í Duda. Út frá þessu getum við leitt út
$\y_k(\x, \w) = \w_k^T \x + w_0$ fyrir fyrstu tvö tilvikin en $\x^T \W \x$ fyrir þriðja tilvikið.

Kafli 5
-------
Logisitic regression, cross enthropy error function

Network training
5.1, 5.2 - 5.3
Backpropagation og svo framvegis.

Mun koma spurning úr kafla 5.3. Skilja líka hvað er að gerast. Setjum inn mælingu og fáum flokk. Þjálfa og beita.

Gradient descent. Errorfallið diffrað mtt einhverrar einstakrar vogtölu, hversu mikil áhrif vogtala hefur áhrif á errorinn.
Munum að við erum að reyna að finna besta staðinn í hæðarlínumyndinni, þar sem errorfallið er sem minnst. Fara hraðast niður í local minimum.

Fór ekki í hessian fylkið, né Newton aðferðina.

Kafli 9
---------
9.1.4.
K-means, Gaussian Mixture Modelling
EM algorithm - pottþétt spyrja um hann, glærur sem fylgja bók

$$p(\x) = \sum_{k=1}^K \pi_k \NormalDist(\x, \mvmean_k, \mvSigma_k)$$

Þrír hlutir sem við gerum með módel
1. Búa til gögn, synthesis
2. Parameter estimation (EM algorithmi)
3. Classify, $\x \rightarrow k$, $\gamma_k(\x)$ eða $\gamma(\z_k) = p(z_k=1| \x)$, jafna 9.13

KNN flokkarnir, parzen Non-parametric aðferðir. Efni í Duda, og kafli 2.5, nátengt kjarnaföllum í kafla 6

Fjórar mismunandi aðferðir til að smíða kjarnafall, skemmtilegast að nota jöfnurnar og púsla saman kjarnaföllum.

Kafli 12
-----------
Bara 12.1
Principal Component Analysis, út á hvað það gengur, maximiza varianceinn í hlutrúminu og minimiza bjögunina.

