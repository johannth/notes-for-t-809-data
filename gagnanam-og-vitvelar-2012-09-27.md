% Notes
% Jóhann Þorvaldur Bergþórsson & Stefanía Bjarney Ólafsdóttir
% 27. September 2012

# Confusion table

We have a classifier and a labeled test set, i.e. two labels $\to$ true, classifier output.

Tabulate this for the test set:

            $c_1$   $c_2$         $c_k$     $E_i$ (misclassification rate)
--------- -------- ------- ----- ------- --------------------------
 class 1   25       3              2       $\frac{5}{30}$
 class 2    4       23             3       $\frac{7}{30}$
 \vdots
 class k    1         0            29      $\frac{1}{30}$

 Average misclassification rate
 $$= \frac{1}{k} \sum_{i=1}^k E_i$$

 Testset misclassification rate
$$=\frac{\# \text{errors}}{\# \text{points}}$$