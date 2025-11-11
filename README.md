# Soft-Margin-SVM
Soft-margin linear SVM trained by (full-batch) gradient descent.  The objective is: J(w) = (1/2)||w||^2 + C * sum_i max(0, 1 - y_i &lt;w, x_i>)     where y_i in {-1, +1}.
