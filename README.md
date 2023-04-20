First of all, we are very happy to be a part of such profound and interesting competition, many thanks to the organizers for their support and to all competitors who selflessly shared their knowledge and insights, @rasmusrse, @rsmits, @solverworld, @edguy99, @iafoss, @pellerphys… Forgive me if I've missed someone, just so many people helped us so much with going forward.  

Many thanks to my great team QuData @synset, @alexz0 and @semenb, you are awesome! We all come from Ukraine, and we wish to thank everyone who is supporting our people now!

This competition was a great opportunity to unite a bit of relevant domain knowledge, as two of us were doing theoretical physics in the past, with the whole team's data science skills and knowledge. 
Taking part in a big astrophysics experiment was novel and exciting, and we hope we made a useful contribution to finding out how the Universe works.

<h2>General architecture</h2>
Our report turned out to be quite large, so a more detailed version of it was posted on the site <a href="https://qudata.com/projects/icecube-neutrino/en/">https://qudata.com/projects/icecube-neutrino/en/</a>
<p> 
The solution has the following structure:
</p>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2Fd27747a4b9d7f2eb613032789cc095a0%2Fcommon_arch.png?generation=1681914442238565&alt=media)

<p> 
"GNN" is a modification of the graph neural network architecture for Neutrino Telescope Event Reconstruction <a href="https://github.com/graphnet-team/graphnet">GraphNet</a>
(public score: <b>0.992</b>)
</p>

<p> 
"Transformer" is a combination of architectures including fully connected blocks, an attention mechanism, and a recurrent layer
(public score: <b>0.995</b>)
</p>

<p> 
"Ensemble" is a neural network that agregates outputs from "GNN" and "Transformer" models and predict direction.
(public score: <b>0.976</b>) 
</p>

<h2>GNN</h2>

<h3>Model</h3>
During the competition, we made many changes to the architecture of the graph neural network (described in detail on the https://qudata.com/projects/icecube-neutrino/en/gnn.html), its final state:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2F000a4894518c3090f4fad6146a865c13%2Fgnn_arch.png?generation=1681914695253186&alt=media)

The input for this model is the graph. Each node corresponds to the pulse, and the features are the features of the pulse combined with the aggregated features of the event. To construct the topology of the graph, the <b>knn</b> algorithm is used. The resulting graph sequentially passes through layers, each of which modifies the feature space and also updates its topology. The outputs of all layers are combined and fed to the layer <b class="norm">Layers MLP</b> in which the number of features is reduced to 256. Then a pooling operation is performed in which the features of all nodes of the graph are aggregated by functions <b class="norm">min, max, mean</b>. At the last node <b class="norm">Classification MLP</b> the resulting embedding is converted into class id and each class is responsible for its own direction.

Below is described our progress that led to this architecture.

<h3>Training and enchancing the model</h2>

<h4>Training base model (1.018 &rarr; 1.012)</h3>

Since we were limited in computational resources, we decided to proceed with retraining the best public GNN model from a notebook <a href="https://www.kaggle.com/code/rasmusrse/graphnet-baseline-submission">GraphNeT Baseline Submission</a> (special thanks to @rasmusrse) which gave LB score <b class="norm">1.018</b>.

We retrained the model on all batches, except for the last one; wich we were used for validation. During training, the learning rate was reduced in steps, at those moments when the validation did not improve for about 100 epochs. As a result, after 956 epochs, the value of the metric dropped to <b>1.0127</b>.

<h4>Adding another layer (1.012 &rarr; 1.007)</h4>

Having a trained network, we tried to add another layer <b class="norm">EdgeConv</b> to it. In order not to learn from scratch, all layers in the new architecture were frozen except for the new one.
&nbsp;

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2Fc2015199cfa3d81a4f028d61458c239d%2Fgnn_plus_layer.png?generation=1681919370984185&alt=media)

In the frozen layers, weights were loaded from the model retrained at the previous stage and training continued.
The model in this mode  learns quickly, and the metric also quickly reaches the level of the model from which the weights are borrowed. Then, unfreezing was performed, the learning rate was reduced, and the entire model was retrained in this mode. After 1077 epochs, we reached the metric <b class="norm">1.007</b>

<h4>Increasing the number of neighbors (1.007 &rarr; 1.003)</h4>

In the original Graphnet library, the number of neighbors for building a graph is chosen as <b>8</b>.
We have tried increasing this value to <b class="norm">16</b>. In the module <b class="norm">EdgeConv</b> the same MLP model is applied to all neighbors, and then the result is summed. Therefore, with an increase in the number of neighbors, the number of network parameters does not change, but at the same time it learns and works twice as slowly.

Thus, we retrained the trained model from the previous stage with a new number of neighbors and after the 1444 epochs, the metric reached a new low of <b class="norm">1.003</b>

<h4>Expanding Layers MLP(1.003 &rarr; 0.9964)</h4>

Since the number of layers was increased, we thought it reasonable that the number of parameters of <b>Layers MLP</b> which receives concatenated input of the outputs of each level, should also be increased. The first layer of the module <b class="norm">Layers MLP</b> was increased from 336 to 2048. Similarly, all levels were frozen, the weights of the model of the previous stage were loaded, and training was continued. After 1150 epochs, the metric dropped to <b>0.9964</b>

<h4>Replacing regression with classification (0.9964 &rarr; 0.9919)</h4>

Studying the best solutions, we paid attention to the notebook <a href="https://www.kaggle.com/code/rsmits/tensorflow-lstm-model-training-tpu">Tensorflow LSTM Model Training TPU</a> (thanks to @rsmits) From it we borrowed the idea to move from a regression problem to a classification one.

The azimuth angle was uniformly divided into 24 bins. The zenith angle was also divided into 24 bins; here we warked with cosine of zenith, since it is the cosine that has a uniform distribution (as is seen from the statistics of all training events).

Accordingly we received a total of <b>24x24=576</b> classes. The the last layer of MLP was increased from [128,3] to [512,576], and the loss-function was changed to <b>CrossEntropyLoss</b>.
We froze the entire model except for the last module and loaded the weights obtained at the previous stage and continued training. After 967 epochs, the metric reached the value of <b>0.9919</b>.
This was the best result that we achieved for a standalone GNN model, and it was then used for
ensembling with other models.

<h3>What didn't help</h3>

<p>
During the competition we tried many other things that did not yield a noticeable improvement
of the metric, or none at all. Some of the things we did:
</p>

<ul>
<li> separately predicting zenith and azimuth angles;
<li> changing the type of convolution to SAGE, GAT;
<li> inserting transformer after Layers MLP;
<li> replacing pooling with RNN (GRU);
<li> inserting a TopKPooling layer;
<li> using 4 or more features to build the event graph.
</ul>

<h3>What have not been done</h3>

<p>
There are also several approaches which we thought over but did not pursued to the end, 
mainly because of lack of time and hardware resources,
to mention just a few:
</p>

<ul>
<li> training the GNN from scratch, with scattering, absorption and DOM-embedding features;
<li> training a classifier with a larger number of bins;
<li> parallel training of transformer and GNN models.
</ul>

<h2>Transformer</h2>

<p>
The model, which we named <a href="https://qudata.com/projects/icecube-neutrino/en/transformer.html">"Transformer"</a>, is a combination of architectures that include fully connected blocks, an attention mechanism, and a recurrent layer.
The typical metric achieved by this model was 0.995.
</p>

<p>
The general architecture of the Transformer model looked like this.
It receives two tensors as input: <b class="norm">(B,T,PF)</b> and <b class="norm">(B,EF)</b>,
where <b class="norm">B</b> is the index of the sample in the minibatch and <b class="norm">T</b> is the index of the pulse number in the sequence.
Each pulse was characterized by <a href="https://qudata.com/projects/icecube-neutrino/en/transformer.html#F"><b class="norm">PF=9</b> features</a>.
The second tensor <b class="norm">(B,EF)</b> characterized the properties of the entire event with <a href="https://qudata.com/projects/icecube-neutrino/en/transformer.html#AF"><b class="norm">EF</b> more 
features</a>.
These features were calculated for all pulses of the event (24 features) and inaddition only for the pulses marked with the auxiliary=0 flag (we'll shorten it to "aux"), 24 more.
</p>

<p>
In some modifications of the architecture, aggregated features with <b class="norm">aux=0</b>, using a fully connected MLP block
(multilayer perceptron) with one hidden layer, were compressed to a smaller dimension and then combined with the pulse feature tensor <b class="norm">(B,T,PF)</b>.
The motivation for compression is related to the fact that the number of event features (24) was significantly greater than the number of more significant pulse features (9).
In other architectures, only features of pulses were used.

</p>

<p>

After concatenation, the resulting tensor <b class="norm">(B,T,F)</b> was fed to  the second MLP block (below the Feature Generator).
Its task was to increase the number of features to <b class="norm">E=128</b>.
The resulting tensor <b class="norm">(B,T,E)</b> was fed to a chain of 10&ndash;12 transformer blocks.
Each pulse, as a result of the mechanism of attention, "interacted" with all the pulses in the sequence.
As a result of the work of the transformer, the pulses received new features in the space of the same dimension.
&nbsp;
 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2Fd1126ccc989cb06af312055fc9185066%2Ftransformer_00.png?generation=1681927827258638&alt=media)

Next, the Integrator block converted the transformer output tensor <b class="norm">(B,T,E)</b> into a tensor of <b class="norm">(B,E')</b> dimensions.
At the initial stage of training, the concatenation of the operation of averaging and obtaining the maximum according to the features of all pulses acted as the Integrator block.
At the final stage, this block was replaced by an RNN layer  integrating the event pulses.
One unidirectional recurrent layer with GRU cells was chosen as RNN.
In this case, the pulses of the event were sorted in reverse order by time.
As a result, the input of the last GRU cell (the output of which was used by the Integrator) received the most significant, that is the first, pulse.
</p>

<p>
The event feature tensor for all pulses and pulses with aux=0 was attached to the <b class="norm">(B,E')</b> tensor obtained after the Integrator.
The resulting tensor <b class="norm">(B,E'+2*EF)</b> was fed to an MLP whose three outputs were Cartesian components
of the direction vector predicted by the regression model.
</p>

<p>
The transformer had a standard architecture with 16 heads in the attention block and trainable weights on skip-connections:
Position embedding was not used (although we experimented with it early on).
Time was one of the features of a pulse, and apparently that was enough.
Numerous, but not final :) experiments were carried out on various modifications of this architecture.
In particular, we studied the version where the final layer was a classifier in 10k directions in space, 
as well as a version in which the transformer consisted of two parallel chains of blocks that were eventually merged together.
The feasibility of such architectural modifications requires further research.
</p>

<a id="dataset"></a>
<h3>Dataset</h3>

<p>
The  transformer training requires the same number of pulses for all examples in the minibatch.
The standard solution to this problem is masking tokens to align sequence lengths.
However, given our computational capabilities, this was a very wasteful method.
So we went the other way.
Every 5 training batches were collected in a pack of 1m examples.
These examples were sorted by sequence length and combined into groups of the same length.
Accordingly, the data were divided into minibatches within each group.
After that, the minibatches of all groups were mixed.
One learning epoch consisted of one pack. Then the next pack was loaded.
</p>
<p>
Another idea that allows you to cope with limited memory was related to the variable size of the minibatch.
The memory requirements increase quadratically with the length of the sequence T.
Therefore, long sequences can be collected into shorter mini-batch, and short sequences into longer batches.
Thus, several hyperparameters were set: T_max - the maximum length of the sequence,
batch_size - batch size for maximum length and batch_max - upper limit of minibatch size for short sequences.
For examples of length T, the minibatch size was determined by the following formula:
<pre class="brush: py">
batch_size = min(int(batch_size * (T_max/T)**2),  batch_max)
</pre>
This led to approximately the same memory consumption for batches with long and short sequences.
</p>

<a id="lr"></a>
<h3>Training implementation</h3>
<p>
Let's show a typical training start schedule (pulses are DOM-aggregated).
An error of the order of 1.020 when learning from scratch is achieved in 60-70 million examples (300 batches, not necessarily new ones).
The total time on the standard T4 card is about 12 hours.
This stage was done at a constant learning rate lr=1e-3 with the Adam optimizer:
</p>
&nbsp;

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2F6bf188353b365fde71a4e1a94fe91b82%2Fexp_large_01.png?generation=1681973550189176&alt=media)

<p>
It took significantly longer to get close to error 1.002.
To achieve it, the learning rate was gradually decreased to 1e-5 (below, the dotted brown line).
At the level of 1.004, there was a transition to large maxima of the sequence lengths.
This immediately reduced the error by 1-1.5 points.
</p>
<p>
Further training required a transition to non-DOM-aggregated pulses and took a few more days on the A100 card.
</p>
<p>
During the training process, we analyzed the value of the weights on various blocks of the model
and  gradient propagation through them. A typical analysis chart looked like this
(see details in <a href="https://qudata.com/projects/icecube-neutrino/en/transformer.html">our report</a>):
</p>
&nbsp;

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2F2cd7550542fa9f05041b1d8266aa9756%2Fexp_large_grad_01.png?generation=1681973578263913&alt=media)

<h2>Ensemble set-up</h2>

<p>

The results of the architectures we trained (<a href="transformer.html">Transformer</a> and <a href="gnn.html">GNN</a>) were not strongly correlated with each other.
This made it possible to form ensembles of models, which gave a significant improvement in the final metric.
In addition, thanks to the remarkably large dataset provided by the organizers, additional progress in the metric could be obtained using the trained ensemble.
The general architecture of the ensemble looked like:
</p>

<p>
<h2>Ensemble Models</h2>
<p>
In each architecture, several models were selected, obtained with different hyperparameters or architecture options.
80 batches were passed through these models, the results of which formed a dataset for the ensemble.
</p>
<p>
Below are the initial data of the selected models. The first column of numbers is the base metric of the competition (mean angular error).
The second two columns are the azimuth error and the module of the zenith error of the model (all validations were carried out on the first 5 batches):
</p>
<pre  style="margin-left:30px">
id   ang_err  az_err  ze_err   model
 0   0.9935   1.035   0.606  | gnn_1
 1   0.9922   1.033   0.595  | gnn_2
 2   0.9963   1.039   0.598  | gnn_3
 3   0.9954   1.018   0.612  | att_1
 4   0.9993   1.022   0.636  | att_2
---------------------------------------------------------------
     0.9846   1.021   0.540  | simple mean 
</pre>
<p>
The criterion for selecting models for the ensemble was the angular error and coefficients of correlations of the angular error between different models:
</p>
<pre style="margin-left:30px">
     0        1        2        3        4
 0            0.959    0.950    0.807    0.781
 1   0.959             0.934    0.820    0.795
 2   0.950    0.940             0.808    0.783
 3   0.807    0.820    0.808             0.940
 4   0.782    0.795    0.783    0.940    
</pre>

<p>
Simply averaging the vectors predicted by each model resulted in a <b class="green">0.9846</b> error.
</p>
<p>
Next, we built an ensemble with trainable weights (the multiplier of each model):
<center>
<b>n</b> = w<sub>1</sub> <b>n</b><sub>1</sub> + ... + w<sub>5</sub> <b>n</b><sub>5</sub>
</center>
This reduced the error to <b class="green">0.9827</b>.
In this case, the regression weights of the models had the form:
</p>

<pre style="margin-left:30px">
gnn_1: 0.934,  gnn_2: 1.346,  gnn_3: 0.753, att_1: 1.466, att_2: 0.477
</pre>

<a id="train"></a>
<h2>Training Ensemble</h2>
<p>
Further advancement of the metric was achieved by a trained ensemble based on a neural network.
Several architectures were considered, including a transformer and adding aggregated event features to the model outputs.
However, they turned out to be no better than conventional MLP with one hidden layer.
The tensor <b class="norm">(B,N,3)</b> was fed at the input of this network, where B is the number of examples in the batch, N is the number of models,
each of which produced three components of the direction vector.
This tensor was converted into dimension <b class="norm">(B,3*N)</b>.
After MLP, its dimension became equal to <b class="norm">(B,3)</b>.

<p>
The number of neurons in the hidden layer ranged from 128 to 2048. However, all these models gave approximately the same results.
For training, the best loss appeared to be the cosine between the predicted and the target vector.
The learning rate was quite high <b class="norm">lr = 1e-3 - 1e-4</b> and the Adam optimizer was used.
</p>
<p>
The best metric, as determined by the validation on the first 5 batches, gave the result <b class="green">0.9796</b>.
Below are some typical learning curves:
</p>
<br>

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1667828%2F2342fe7cc1946eaaee3ca0f42d8a4339%2Fensemble_train.png?generation=1681973858282499&alt=media)

<h3>Source code</h3>
<p>
The source code of all our solutions can be found on <a href="https://github.com/QuDataAI/IceCube">https://github.com/QuDataAI/IceCube</a>
Below are links to modules:

<ul>
<li> <a href="https://github.com/QuDataAI/IceCube/">IceCube-Dataset-kaggle</a> - сreating a dataset for transformer;
<li> <a href="https://github.com/QuDataAI/IceCube/">IceCube-Transformer-kaggle</a> - transformer train;
<li> <a href="https://github.com/QuDataAI/IceCube/kaggle_2023/IceCube_GNN_train_colab.ipynb">IceCube-GNN-train-colab</a> - GNN train;
<li> <a href="https://github.com/QuDataAI/IceCube/">IceCube-Ensemble-train</a> - Ensemble train;
<li> <a href="https://www.kaggle.com/andreybeyn/icecube-ensemble-submit">IceCube-Ensemble-submit</a> - Ensemble submission;
</ul>

<a id="not_done"></a> 
<h3>What have not been done</h3>

<p>
There are also several approaches which we thought over but did not pursued to the end, 
mainly because of lack of time and hardware resources,
to mention just a few:
</p>

<ul>
<li> Training the GNN from scratch, with scattering, absorption and DOM-embedding features;
<li> Training a GNN classifier with a larger number of bins;
<li> Parallel training of transformer and GNN models.
<li> Use a bidirectional recurrent layer as integrator in transformer.
<li> Use the transformer pre-trained on the regression task to detect noise pulses, for their subsequent filtering.
<li> Replace the regressor with a classifier in transformer, for a large set of directions (points on a sphere).
      A similar model got to error 1.012, but there was not enough time to retrain it and include it in the ensemble.
<li> Train a model on a subset of the data (for example, single-string events, two-string events, etc.) by obtaining a set of models that specialize in specific event patterns.
<li> Add position embedding in transformer.

</ul>

<h2>Conclusions</h2>

Thanks to the organizers for large dataset, without which we would not have been able to train complicated architectures and combine them into an ensemble.
