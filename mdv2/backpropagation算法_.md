# backpropagation算法原理
  
-------------------------
  
> Backpropagation核心解决的问题: ∂C/∂w 和 ∂C/∂b 的计算, 针对cost函数C
  
![](2018-05-08-20-50-22.png )
> <img src="https://latex.codecogs.com/gif.latex?&#x5C;omega_{24}^{3}:"/>表示第从第(3-1)层的的第4个神经元到第3层的第2个神经元的权重weight 
  
![](2018-05-08-20-51-28.png )
> <img src="https://latex.codecogs.com/gif.latex?b_{3}^{2}:"/>表示第2层的第3个神经元的偏向bais
-----
## 正向传播
  
  
###### 公式:<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha_{j}^{l}=&#x5C;sigma(&#x5C;sum_{k}&#x5C;omega_{jk}^{l}&#x5C;alpha_{k}^{l-1}+b_{j}^{l})"/></p>  
  
  
  
>分为两步：
>1. <img src="https://latex.codecogs.com/gif.latex?&#x5C;omega&#x5C;alpha+b"/> 加权求和
>2. 对加权求和整体使用sigmoid函数求出下层的输出值 
##### 数据结构
  
> 对于<font color='red'>每一层(l)</font>,定义一个权重矩阵     (weight matrix):<img src="https://latex.codecogs.com/gif.latex?&#x5C;omega^{l}"/>，这个权重矩阵包含当前层的所有神经元到前一层的所有神经元的权重
>> <img src="https://latex.codecogs.com/gif.latex?&#x5C;omega_{jk}^{l}:"/>表示第从第l层的的第j个神经元到第l-1层的第k个神经元的权重weight 
  
>对于每一层(l),定义一个偏向向量(bais vector):<img src="https://latex.codecogs.com/gif.latex?b^{l}"/>
>> <img src="https://latex.codecogs.com/gif.latex?b_{j}^{l}"/>则表示l层的第k个神经元的bais
  
>同理，对于<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha"/>：l层的神经元向量<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha^{l}"/>,每个神经元的值<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha_{j}^{l}"/>
  
  
> Vector a function: <img src="https://latex.codecogs.com/gif.latex?&#x5C;sigma(&#x5C;upsilon)_{j}%20=%20&#x5C;sigma(&#x5C;upsilon_{j})"/>
> 例如：<img src="https://latex.codecogs.com/gif.latex?f(x)=x^{2}"/>
> <img src="https://latex.codecogs.com/gif.latex?f(&#x5C;begin{bmatrix}2&#x5C;&#x5C;3&#x5C;end{bmatrix})=&#x5C;begin{bmatrix}f(2)&#x5C;&#x5C;f(3)&#x5C;end{bmatrix}=&#x5C;begin{bmatrix}%204&#x5C;&#x5C;9%20&#x5C;end{bmatrix}"/>
>则,可以由l层的每一个元素的计算公式可以退出该层矩阵运算的公式:<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha_{j}^{l}=&#x5C;sigma(&#x5C;sum_{k}&#x5C;omega_{jk}^{l}&#x5C;alpha_{k}^{l-1}+b_{j}^{l})=&gt;&#x5C;alpha^{l}=&#x5C;sigma(&#x5C;omega^{l}&#x5C;alpha^{l-1}+b^{l})"/></p>  
  
>其中<img src="https://latex.codecogs.com/gif.latex?&#x5C;sum_{k}&#x5C;omega_{jk}^{l}&#x5C;alpha_{k}^{l-1}"/>可以用<img src="https://latex.codecogs.com/gif.latex?&#x5C;omega_{j}^{l}&#x5C;alpha_{k}^{l-1}"/>表示,即第l层权重矩阵的第j个行向量与前一层神经元的列向量进行内积的结果,令<img src="https://latex.codecogs.com/gif.latex?z^{l}=&#x5C;omega^{l}&#x5C;alpha^{l-1}+b^{l}"/>，则<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha^{l}=&#x5C;sigma(z^{l})"/>。
>简单点理解：对于每层正向传播,每层的值(向量形式)就是当前层的权重矩阵与上一层的值(向量)相乘再加上当前层的偏向(向量),然后统一使用sigmoid函数转化
--------------
  
## 关于Cost函数的两个假设:
  
> 二次Cost函数
><p align="center"><img src="https://latex.codecogs.com/gif.latex?C=&#x5C;frac{1}{2n}&#x5C;sum_{x}||y(x)-&#x5C;alpha^{L}(x)||^{2}"/></p>  
  
>其中<img src="https://latex.codecogs.com/gif.latex?&#x5C;alpha^{L}(x)"/>表示输出层的真实值所组成的向量,<img src="https://latex.codecogs.com/gif.latex?x"/>表示训练实例，n表示输入实例的个数。
  
> 1. 第一个假设是成本函数可以写成单个训练样例<img src="https://latex.codecogs.com/gif.latex?x"/>的成本函数<img src="https://latex.codecogs.com/gif.latex?C_{x}"/>的平均值<img src="https://latex.codecogs.com/gif.latex?C=&#x5C;frac{1}{n}&#x5C;sum_{x}C_{x}"/>。二次成本函数就是这种情况，单个训练样例的Cost函数为<img src="https://latex.codecogs.com/gif.latex?C_x%20=%20&#x5C;frac%20{1}%20{2}%20&#x5C;|%20y-a%20^%20L%20&#x5C;|%20^%202"/>
>2. 第二个假设是它可以写​​成神经网络输出的函数：
>![](2018-05-09-19-22-34.png )
二次Cost函数满足这个要求，因为单个训练样例x的二次Cost可写为<p align="center"><img src="https://latex.codecogs.com/gif.latex?C%20=%20&#x5C;frac{1}{2}%20&#x5C;|y-a^L&#x5C;|^2%20=%20&#x5C;frac{1}{2}%20&#x5C;sum_j%20(y_j-a^L_j)^2"/></p>  
  
>###### 介绍一个后面需要用到的公式
>The Hadamard product, s⊙t,向量对应元素相乘:
><img src="https://latex.codecogs.com/gif.latex?&#x5C;begin{bmatrix}1&#x5C;&#x5C;2&#x5C;end{bmatrix}⊙&#x5C;begin{bmatrix}3&#x5C;&#x5C;4&#x5C;end{bmatrix}=&#x5C;begin{bmatrix}%201*3&#x5C;&#x5C;2*4%20&#x5C;end{bmatrix}=&#x5C;begin{bmatrix}3&#x5C;&#x5C;8&#x5C;end{bmatrix}"/>
--------------
  
## backpropagation4个重要公式
  
![](2018-05-09-19-08-22.png )
> 反向传播是关于如何改变网络中的权重和偏差来改变成本函数，这意味着需要计算偏导数<img src="https://latex.codecogs.com/gif.latex?&#x5C;partial%20C&#x2F;&#x5C;partial&#x5C;omega^{l}_{jk}"/> 和 <img src="https://latex.codecogs.com/gif.latex?&#x5C;partial%20C%20&#x2F;%20&#x5C;partial%20b%20^%20l_j"/>。但为了计算这些，我们首先引入一个中间量，<img src="https://latex.codecogs.com/gif.latex?&#x5C;delta^l_j"/>，我们称之为在l层的第j个神经元的error。反向传播计算每一层的<img src="https://latex.codecogs.com/gif.latex?&#x5C;delta^l_j"/>，然后将<img src="https://latex.codecogs.com/gif.latex?&#x5C;delta^l_j"/>与<img src="https://latex.codecogs.com/gif.latex?&#x5C;partial%20C&#x2F;&#x5C;partial&#x5C;omega^{l}_{jk}"/> 和 <img src="https://latex.codecogs.com/gif.latex?&#x5C;partial%20C%20&#x2F;%20&#x5C;partial%20b%20^%20l_j"/>关联起来。
>
>为了理解错误是如何定义的，想象我们的神经网络中存在一个恶魔：
  
![](http://neuralnetworksanddeeplearning.com/images/tikz19.png )
>恶魔对第l层的第j个神经元添加一个变化量<img src="https://latex.codecogs.com/gif.latex?&#x5C;Delta%20z%20^%20l_j"/>,该神经元输出就变成<img src="https://latex.codecogs.com/gif.latex?&#x5C;sigma(z%20^%20l_j%20+%20&#x5C;Delta%20z%20^%20l_j)"/>。这种变化通过网络中的后续层传播，最终导致整体Cost的变化<img src="https://latex.codecogs.com/gif.latex?&#x5C;frac%20{&#x5C;partial%20C}%20{&#x5C;partial%20z%20^%20l_j}%20&#x5C;Delta%20z%20^%20l_j"/>（简单的高数知识）。
> 如果这个恶魔是一个好人，它想要帮我们优化Cost，他会尝试一个更小的<img src="https://latex.codecogs.com/gif.latex?&#x5C;Delta%20z%20^%20l_j"/>使得损失函数更小。假设<img src="https://latex.codecogs.com/gif.latex?&#x5C;frac%20{&#x5C;partial%20C}%20{&#x5C;partial%20z%20^%20l_j}"/>是一个很大的值(不管正负)。然后恶魔通过选择与<img src="https://latex.codecogs.com/gif.latex?&#x5C;frac%20{&#x5C;partial%20C}%20{&#x5C;partial%20z%20^%20l_j}"/>有相反的符号的 <img src="https://latex.codecogs.com/gif.latex?&#x5C;Delta%20z%20^%20l_j"/>来降低Cost。相反，如果 <img src="https://latex.codecogs.com/gif.latex?&#x5C;frac%20{&#x5C;partial%20C}%20{&#x5C;partial%20z%20^%20l_j}"/> 接近于零，那么恶魔通过干扰加权输入<img src="https://latex.codecogs.com/gif.latex?z_{j}^{l}"/>就几乎不能改变Cost,此时，这个神经元已经非常接近最优(再如何优化也不能改变Cost)。<font color='red'>所以这里把<img src="https://latex.codecogs.com/gif.latex?&#x5C;frac%20{&#x5C;partial%20C}%20{&#x5C;partial%20z%20^%20l_j}"/> 定义为神经元error的度量</font>。
>于是定义在l层的第j个神经元的error<img src="https://latex.codecogs.com/gif.latex?:&#x5C;delta_{l}^{j}"/>
  
BP1
> 我们定义在输出层(L)的第j个神经元的error的方程为:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;delta^L_j%20=%20&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20a^L_j}%20&#x5C;sigma&#x27;(z^L_j)%20%20%20%20%20(BP1)"/></p>  
  
> ###### 解释:其中<img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20a^{L}_{j}}"/>这部分衡量Cost相对于第j个神经元activation的输出的变化率,<img src="https://latex.codecogs.com/gif.latex?&#x5C;sigma&#x27;(z^L_j)"/>这部分衡量activation方程相对于中间变量<img src="https://latex.codecogs.com/gif.latex?z^{L}_{j}"/>的变化率
> 转化为矩阵的表达形式
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;delta%20^%20L%20=%20&#x5C;nabla_a%20C%20&#x5C;odot%20&#x5C;sigma&#x27;(z%20^%20L)"/></p>  
  
可以认为 <img src="https://latex.codecogs.com/gif.latex?&#x5C;nabla_a%20C"/>表示C相对于输出activation的变化率，根据上面定义的2次Cost方程，输出层的error可以写成:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;delta%20^%20L%20=%20(a%20^%20L-y)&#x5C;odot%20&#x5C;sigma&#x27;(z%20^%20L)"/></p>  
  
  
BP2
> 因为下一层的error的变化会引起当前层error的变化,当前层(l层)的error变化方程为(error传递公式):
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;delta%20^%20l%20=((w%20^%20{l%20+%201})^%20T%20&#x5C;delta^{l%20+%201})&#x5C;odot%20&#x5C;sigma&#x27;(z%20^%20l)(BP2)"/></p>  
  
其中<img src="https://latex.codecogs.com/gif.latex?(w%20^%20{1%20+%201})^%20T"/>是第<img src="https://latex.codecogs.com/gif.latex?(1%20+%201)^{&#x5C;rm%20th}"/>层的权重矩阵<img src="https://latex.codecogs.com/gif.latex?w%20^%20{l%20+%201}"/>的转置。第<img src="https://latex.codecogs.com/gif.latex?(l%20+%201)%20^%20{&#x5C;rm%20th}"/>层处的error <img src="https://latex.codecogs.com/gif.latex?&#x5C;delta%20^%20{l%20+%201}"/>乘以<img src="https://latex.codecogs.com/gif.latex?(l%20+%201)%20^%20{&#x5C;rm%20th}"/>权重矩阵的转置<img src="https://latex.codecogs.com/gif.latex?(w%20^%20{l%20+%201})^%20T"/>时，我们可以直观地认为网络向前传递error。然后<img src="https://latex.codecogs.com/gif.latex?&#x5C;odot%20&#x5C;sigma&#x27;(z%20^%20l)"/>,可以算出前一层的error。
交替使用可以算出神经网络的所有层的error
  
BP3
> Cost对偏向求偏导：
> <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20b^l_j}%20=%20%20&#x5C;delta^l_j%20(BP3)"/></p>  
  
> 写成向量形式:
> <p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20b}%20=%20&#x5C;delta"/></p>  
  
  
BP4
>Cost对权重求偏导:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20w^l_{jk}}%20=%20a^{l-1}_k%20&#x5C;delta^l_j(BP4)"/></p>  
  
写成矩阵形式:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?&#x5C;frac{&#x5C;partial%20C}{&#x5C;partial%20w}%20=%20a_{&#x5C;rm%20in}%20&#x5C;delta_{&#x5C;rm%20out}"/></p>  
  
  
  
  
[公式证明参考](http://neuralnetworksanddeeplearning.com/chap2.html#proof_of_the_four_fundamental_equations_(optional ))
  