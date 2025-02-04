<!--

author:   Rose Hartman
email:    hartmanr1@chop.edu
version:  1.0.1
current_version_description: Initial version
module_type: standard
docs_version: 1.0.0
language: en
narrator: UK English Female
mode: Textbook

title: Demystifying Machine Learning

comment:  An approachable and practical introduction to machine learning for biomedical researchers.

long_description: If you're curious about machine learning and whether or not it could be useful to you in your work, this is for you. It provides a high-level overview of machine learning techniques with an emphasis on applications in biomedical research. This module covers the what and the why of machine learning only, not the how -- it doesn't include instructions or code for running models, just background to help you think about machine learning in research.

estimated_time_in_minutes: 60

@pre_reqs

This module assumes learners have been exposed to introductory statistics, like the distinction between [continuous and discrete variables](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-and-continuous-random-variables).
There are no coding exercises, and no programming experience is required.

@end

@learning_objectives  

After completion of this module, learners will be able to:

- list at least three potential applications of machine learning in biomedical science
- describe three different statistical problems models can address and how they differ (e.g. prediction, anomaly detection, clustering, dimension reduction)
- describe some potential pitfalls of machine learning and big data

@end

good_first_module: true
collection: demystifying, machine_learning
coding_required: false

@sets_you_up_for

- bias_variance_tradeoff

@end

@depends_on_knowledge_available_in

@end

@version_history
No previous versions.
@end

import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros.md
-->

# Demystifying Machine Learning

@overview

## What exactly is machine learning?

Machine learning is a general term used to describe a range of different techniques to find and use patterns in data, especially large and messy data.
In particular, it refers to **algorithms whose performance improves as they are given more data** -- that's why it's called "learning".

Machine learning is a complex field that shares a lot of overlap with other topics, which can make it difficult to know what machine learning even is, let alone how to do it.

We'll start here by talking about how machine learning differs from other closely related fields like statistics and artificial intelligence.

![One stick figure holding a large paddle stands on what looks like a compost pile with Greek letters, numbers, and matrices mixed in. There is a large data funnel on one side and an answers box at the other. A second person approaches and says, "This is your machine learning system?" The first person responds, "Yup! You pour the data into this big pile of linear algebra, then collect answers on the other side." The second asks, "What if the answers are wrong?" The first answers, "Just stir the pile until they start looking right."](https://imgs.xkcd.com/comics/machine_learning.png "[Machine Learning](https://xkcd.com/1838/) comic by xkcd, [CC BY-NC 2.5](https://xkcd.com/license.html).")

### Machine learning vs. artificial intelligence

Artificial intelligence (AI) is a broader category that includes things like machine learning, but also other tools like computer vision and neural networks.

AI is any computer system that seeks to mimic (or out-perform) human capabilities.
Machine learning is one example of this -- extracting patterns and making predictions from data is a lot of what human cognition is all about -- but not all AI can be described as machine learning.

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

To read more on this topic, check out this [blog post on the distinction between AI and machine learning](https://ai.engineering.columbia.edu/ai-vs-machine-learning/).

</div>

It's common in some publications to see machine learning approaches described as "artificial intelligence" rather than machine learning per se.
That's fine, and it's technically correct -- all machine learning models are examples of AI.

And some AI systems may include machine learning components in addition to other aspects of AI.
For example, [driverless cars rely on machine learning algorithms to process their environment and make predictions](https://www.bu.edu/articles/2021/self-taught-self-driving-cars/), but driving a car safely requires other tasks, like route planning, that are still examples of AI without really being machine learning.

### Machine learning vs. statistics

Machine learning is often based on classic statistical models (especially regression), but machine learning differs from more classic statistical tests in a few important ways:

The way you solve the equation
---

Statistical tests are (usually) **computationally tractable**, meaning given enough time and knowledge of matrix algebra you could theoretically work out the results with a pencil and paper.
In contrast, machine learning relies on tools like [gradient descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) to "brute force" solutions by **iteratively fitting solutions** and checking them to identify the best one.
You need a computer to do machine learning, but you don't necessarily need one for statistics.

The kind of data you apply it to
---

Most statistical tests are intended for relatively **small datasets** (no more than a dozen or so variables, and usually no more than hundreds or thousands of observations).
It's possible to run many classic statistical tests on large datasets, but that's not the use they were designed for.
Machine learning algorithms, on the other hand, were built for **large and messy data**.
When you have a lot of variables to analyze and a lot of observations, machine learning may provide solutions that are quicker, more robust, and actually easier to interpret than more traditional statistical tests.

Moreover, many machine learning approaches actually won't work if your dataset is too small.  If you don't have enough information in the data, your machine learning algorithm may not be able to **converge**, and you'll get no results at all.

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

What is convergence?  When an algorithm converges, it is closing in on a potential solution (a reasonably accurate prediction, for example) as it iteratively changes its models in attempts to improve.  If you remember playing games as a child where a player would say "warmer... warmer... hot!" as you got closer to the correct location or guess, that's what convergence is like.  The algorithm tweaks its guess by making changes in a given direction which consistently leads to a better outcome, making it easy to tell when it's on the right track and which direction to go in to get an even better guess.

Convergence fails when changes to a machine learning model instead lead to chaotic outcomes, like very poor predictions followed by good ones followed by terrible ones followed by great ones.  It means there's no clear direction to move in: "Warmer! Cold!  Colder! Hot!".  

</div>

Inference vs. prediction: A subtle distinction
---

Statistical tests are generally designed for **inference** -- you measure relationships in your sample in order to infer something about the relationships in a larger population.
Machine learning models are designed for **prediction** and **generalization** -- you build a model on your sample with the intention of predicting outcomes in new data and/or generalizing patterns to new data.

This is a tricky distinction because, in practice, an analyst will often use a statistical model to infer something about the population of interest with the intention of making a prediction.

For example, you might assess the relationship between [measurements taken during a prenatal ultrasound and the eventual weight of babies at birth](https://pubmed.ncbi.nlm.nih.gov/3881966/) because you want to be able to say for a particular patient, "Given the measurements we're seeing for the fetus today, I'd say she's at the 50th percentile for weight right now."
You can't measure the fetus's weight directly, but you're making a prediction about it based on what you know about the relationship between weight and, for example, sonographic measurements of femur length.
We frequently make predictions based on the results of inferential statistical tests.

The key difference is that statistical methods start with inference (which you might then logically extend to make a prediction), but machine learning can start directly with prediction.
Machine learning does not necessarily infer anything about the population.
For example, we may eventually replace manual ultrasound measurements with a machine learning algorithm that can produce a fetal weight estimate from the images.
Some machine learning approaches would produce those predictions without actually telling us anything interpretable at all about how information in the ultrasound relates to fetal weight.

<div class = "care">
<b style="color: rgb(var(--color-highlight));">A little encouragement...</b><br>

If you read the section above and thought "I'm still not sure I understand how to tell machine learning from other kinds of statistics", don't worry; it's a distinction with a lot of grey area.
As the authors in this [article in Nature Methods](https://www.nature.com/articles/nmeth.4642) note, the boundary is not always clear:

>The boundary between statistical inference and ML is subject to debate -- some methods fall squarely into one or the other domain, but many are used in both.

</div>



## Different questions need different kinds of models

Of course, not every research question is a good fit for machine learning, and within machine learning there are many different kinds of models to choose from.

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

For a more in-depth discussion of how to select a machine learning model, work through [Google's free course on how to identify and frame machine learning problems](https://developers.google.com/machine-learning/problem-framing).

</div>

To give you a sense of what kinds of problems researchers solve with machine learning, we'll include a quick overview here of some of the more popular techniques as well as some examples of how they've been used in the biomedical and health sciences.

**This is nowhere near a complete list!**
To keep learning about more kinds of machine learning models, check out the links in the [Additional Resources](#additional-resources) section.


<div class = "care">
<b style="color: rgb(var(--color-highlight));">A little encouragement...</b><br>

As in many fields, machine learning involves a lot of technical language, some of which is unclear, redundant, or downright confusing.
For example:

**Outcome** variables are also called **response variables**, **dependent variables**, or **labels**.

**Input** variables are also called **predictors**, **features**, **independent variables**, or even just **variables**.

To make matters worse, sometimes the same words are used to mean different things in different subfields.
If you find yourself stumbling on vocabulary as you read about machine learning, know you're not alone!

</div>

### Supervised vs. unsupervised machine learning

Within machine learning, many techniques can be described as one of two basic kinds of models: **supervised** and **unsupervised**.

From [Google's excellent introductory course on machine learning](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)):

> [Supervised learning](https://developers.google.com/machine-learning/glossary#supervised-machine-learning) models can make predictions after seeing lots of data with the correct answers and then discovering the connections between the elements in the data that produce the correct answers. This is like a student learning new material by studying old exams that contain both questions and answers. Once the student has trained on enough old exams, the student is well prepared to take a new exam. These ML systems are “supervised” in the sense that a human gives the ML system data with the known correct results.

For example, a model that predicts which patients will need to be rehospitalized within a month after discharge based on previous data from electronic health records for patients who did or didn't need to be rehospitalized would be a supervised model.
Supervised models require [labeled data](https://developers.google.com/machine-learning/glossary#labeled-example) to train on -- data that has the outcome included.

Again, from Google's machine learning course:

> [Unsupervised learning](https://developers.google.com/machine-learning/glossary#unsupervised-machine-learning) models make predictions by being given data that does not contain any correct answers. An unsupervised learning model's goal is to identify meaningful patterns among the data. In other words, the model has no hints on how to categorize each piece of data, but instead it must infer its own rules.

A model identifying clusters within cancer cell lines would be an unsupervised model.
Unsupervised models train on unlabeled data.

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

Although supervised and unsupervised machine learning cover a lot of techniques -- especially the more common ones used in research -- there are plenty of machine learning models that fall into neither category.
One important example is [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning).

</div>

### Prediction

The goal of prediction models is to predict an outcome as accurately as possible.
Prediction models are usually supervised, meaning the training data includes the outcome of interest.

For example, consider [this article on medical image analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0037245).
The authors used crowd sourcing to label a huge dataset of blood smear images as either infected with malaria or not, and then used that labeled dataset to train a machine learning algorithm to correctly identify infected cells in new blood smear images.
The outcome in this case is whether or not the cell is infected with malaria, and the input data are the images.

Prediction models can be **regression models**, when the outcome you want to predict is a continuous variable, or **classification models**, when the outcome is a discrete variable.
Within classification models, there's a distinction between binary classification (when the outcome has only two possible values, like yes or no) and multi-class classification (when the outcome has more than two possible values).
The article cited above is an example of a binary classification model.

### Anomaly detection

The goal of anomaly detection is to identify observations that deviate from the expected pattern in the rest of the data, also called [outliers](https://en.wikipedia.org/wiki/Outlier).
Anomaly detection can be supervised or unsupervised.

In supervised anomaly detection, you train the model on labeled data, where all of the observations are marked as either normal or anomalous.
Unsupervised anomaly detection identifies outliers from unlabeled data.
Typically, the model (sometimes called a detector) is trained on a dataset of only normal observations and is designed to detect any new observations that deviate from what it learned in the training data.

Because by definition anomalies are much less common than typical observations, creating a training dataset with enough labeled anomalies to train a good supervised model is hard.
Unsupervised anomaly detection is therefore more often used than supervised anomaly detection.

For an example of anomaly detection in action, check out this article on [outlier detection for patient monitoring and alerting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3567774/).
The authors created a machine learning algorithm that was trained on the electronic health records of post-cardiac surgical patients with the intention of flagging any anomalies in patient-management decisions.
By detecting decisions that deviate from typical patient care, the system can alert providers, potentially reducing medical errors.


### Clustering

The goal of clustering is to **group together observations** that show a similar pattern of response across variables.
Clustering models are unsupervised, because there is no outcome variable present in the training data.

For example, if you have a dataset of a variety of risk factors in adolescents, you may want to run a clustering analysis to see what typical profiles emerge (see [Walsh et al., 2020](https://doi.org/10.1016/j.jadohealth.2020.02.012)).
You might see one group of adolescents who are more likely to report both substance use and early sex, and another group who report insufficient nutrition, and so on.

Another common application of clustering algorithms is [identifying putative cell types in single-cell RNA sequencing](https://www.sciencedirect.com/science/article/pii/S0098299717300493?via%3Dihub).

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

Two common approaches for clustering analysis are [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) and [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering), but there are many more techniques available.

For a general overview of clustering algorithms, see the [Google course on clustering](https://developers.google.com/machine-learning/clustering/clustering-algorithms).

For an overview of clustering algorithms specifically as applied in single cell RNA sequencing, see [Duò, Robinson and Soneson, 2020](https://f1000research.com/articles/7-1141/v3).

</div>

### Dimension reduction

The goal of dimension reduction is to **summarize complex data**, so it can be represented with **fewer variables** (fewer dimensions).
For example, if you have a dataset with 100 variables, you might use dimension reduction to get that down to just 10 variables that still capture nearly all of the information from the original data.
Like clustering, dimension reduction models are unsupervised.

There are two main ways to reduce the number of variables in your data: drop variables that you don't need, or summarize a lot of variables with fewer variables.
Dropping unnecessary variables is called [feature selection](https://en.wikipedia.org/wiki/Feature_selection), and is built into some machine learning approaches such as [lasso regression](https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6062a083acbfe82c7195b27d/1617076404560/ISLR%2BSeventh%2BPrinting.pdf#page=226).
There are many approaches for summarizing across multiple variables, but the most common approach is probably Principal Component Analysis (PCA).

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

For more detail, check out [StatQuest](https://statquest.org/about/)'s great video on Principal Component Analysis:

!?[StatQuest: Principal Component Analysis (PCA), Step-by-Step.](https://www.youtube.com/watch?v=FgakZw6K1QQ)

</div>

Dimension reduction is sometimes used as an initial step in machine learning, to simplify the data before running it through another kind of analysis.
For example, because sequencing datasets are very complex, dimension reduction is an important step in RNA sequencing analysis (see [Risso et al., 2018](https://www.nature.com/articles/s41467-017-02554-5.pdf)).
This is especially helpful when you have so many variables that your data are computationally difficult to analyze.

<div class = "cool-fact">
<b style="color: rgb(var(--color-highlight));">Did you know?</b><br>

Having data that are too complex to analyze is sometimes called the "[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)".

</div>

## A word of caution

Big data is often also messy data, and that can make it tricky to work with statistically.
Moreover, the complexity of some machine learning models can create new problems, especially with respect to explainability and fairness. 

Machine learning is an important and powerful tool, but it has its shortcomings.

![Panel 1: Two people talking, one says "Our field has been struggling with this problem for years" as a third person approaches. Panel 2: Third person, holding up a laptop, says, "Struggle no more! I'm here to solve it with algorithms!" Panel 3: Third person working on laptop while the other two watch. Panel 4: Six months later, third person says "Wow, this problem is really hard." First person says "You don't say."](https://imgs.xkcd.com/comics/here_to_help_2x.png "[Here to Help](https://xkcd.com/1831/) comic by xkcd, [CC BY-NC 2.5](https://xkcd.com/license.html).")


### Data quality

[Garbage in, garbage out](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out) is such a common saying in mathematics and computer science that it's often simply abbreviated GIGO.
The idea is that if your inputs are bad, no analysis in the world will be able to produce valuable output from them.

Data quality is of particular concern in machine learning because very large datasets can sometimes make it harder to spot problems with data quality.
If you have ten variables, you might notice that one of them looks like it was miscoded just by glancing at the data file, but what if you have hundreds of variables?
And millions of observations?

To make matters even worse, the models themselves are sometimes complex enough that even the people running the analysis might not understand exactly how the inputs and outputs are related -- that makes it harder to catch data quality problems by noticing confusing results.

**So what do you do?**

Don't skimp on exploratory data analysis.
Check that distributions look like what you would expect, examine outliers, understand your missing data.
If there is grouping structure in your data, run all of your exploratory analyses both within and across groups.
If anything seems off, pursue it until you understand.

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

For a detailed practical guide to data exploration in machine learning, check out the [Google guide to good data analysis](https://developers.google.com/machine-learning/guides/good-data-analysis).

</div>

Also, crucially, if you're using an existing dataset, be sure to read any accompanying documentation!
There may be known quality issues in the data, and you can save yourself a lot of headaches by learning as much as you can **about** the data before you try to learn anything **from** the data.

### Model explainability

In machine learning, especially on big data, sometimes models are so complex in the way they combine and perform math on features that it's not clear in human terms why a particular prediction was made. 
This feels very different from simple statistical inference, which can usually be described in a sentence or two: "as number of school days lost to illness rises, patient-reported quality of life scores fall in a linearly inverse relationship."  

Particularly complex models which are hard to explain because they use many steps or do advanced mathematical techniques are called "black boxes". 
They can be very accurate for prediction and can do a good job of finding relationships that humans would miss, as they combine hundreds of data points on millions of observations. 
An example of a black box algorithm might be credit card anomaly detection. 
If you've ever had someone say to you, "well, it's not clear why we thought your credit card was stolen, but there was just something about your purchases and travel that made us automatically block your card," you have experienced the unexplainability of a black box algorithm.

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

Explainability is less important in some fields and more important in others. 
If you are creating models that require human buy-in or are aimed at helping people change behaviors to reduce risk, explainability is paramount.  

</div>

For example, consider a hypothetical black box model of cancer risk, which takes hundreds of variables including where people lived at different points in their life, jobs they've had, lab values, marital status, diet and exercise habits, genomic data, known heavy metal exposures, etc. 
This is a staggeringly complex model that can accurately predict solid tumor risk, but because it combines and recombines data and uses strategies that are hard to describe, it's not explainable. 
That means that if a patient asks, "will reducing my red meat consumption lower my cancer risk?" there's no way to know how that particular variable affects the final model outcome.

Black box models can also be unfair in ways that are obscured, and may produce results that are biased in terms of race, gender, or other category of human bias. 
Demonstrating and ameliorating bias in black box models can be tricky because of the number of potentially biased features and the complex algorithm that processes those features.

Explainable machine learning (also known as explainable AI) can be achieved by:

 * Using algorithms that are simpler and more intuitive (for example, using decision trees instead of neural networks)
 * Simulating the results of changing a single feature to measure how that feature influences the prediction
 * Using software or code libraries that can show the relative salience or weight of various features

### Algorithmic fairness

Many of the large datasets that might be analyzed with machine learning reflect the unfair and immoral inequalities inherent in our society; models trained on data with that kind of bias will reproduce that bias in their predictions and suggestions.
That means that it's very easy to unintentionally create machine learning systems that enforce racist/sexist/ableist reasoning, and that may have terrible consequences if applied in the real world.

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

To learn more about one particularly salient example of this problem, read ["Dissecting racial bias in an algorithm used to manage the health of populations"](https://www.science.org/doi/full/10.1126/science.aax2342) by Ziad Obermeyer and colleagues. From the abstract:

> The U.S. health care system uses commercial algorithms to guide health decisions. Obermeyer et al. find evidence of racial bias in one widely used algorithm, such that Black patients assigned the same level of risk by the algorithm are sicker than White patients (see the Perspective by Benjamin). The authors estimated that this racial bias reduces the number of Black patients identified for extra care by more than half. Bias occurs because the algorithm uses health costs as a proxy for health needs. Less money is spent on Black patients who have the same level of need, and the algorithm thus falsely concludes that Black patients are healthier than equally sick White patients. Reformulating the algorithm so that it no longer uses costs as a proxy for needs eliminates the racial bias in predicting who needs extra care.

You can also listen to the first author, [Ziad Obermeyer, talking about this paper at a workshop on algorithmic fairness](https://www.youtube.com/watch?v=u34e9CYIMZs).

</div>

Mathematician Cathy O'Neil has done a lot to bring public attention to issues of bias and unfairness in machine learning algorithms used in the sectors like education, finance, and the justice system.
For an excellent overview of the problem, watch her TED talk:

!?[Cathy O'Neil's TED Talk, "The era of blind faith in big data must end".](https://www.youtube.com/watch?v=_2u_eHHzRto)

**What should you do about unintended biases in your own algorithms?**

As both O'Neil and Obermeyer emphasize, algorithms can be interrogated.
In addition to assessing how accurate a model is, you can and should look for biases in your results.

For example, if you expect that your model's predictions shouldn't be systematically different for patients on [Medicaid](https://www.hhs.gov/answers/medicare-and-medicaid/who-is-eligible-for-medicaid/index.html) vs. private insurance, then check the output to see if that's the case.


## Quiz

Which of the following would be examples of machine learning? Select all that apply.

[[X]] Using ultrasound images, automatically flag potentially high-risk patient files for closer inspection by a clinician
[[ ]] Estimate the difference in average number of ER visits per year for patients with and without health insurance
[[X]] Based on a large sample of electronic health records, build a model to give the probability that a given patient will be readmitted within a week of hospital discharge
[[X]] Using data from electronic health records, identify meaningful clusters of patients with similar symptoms
[[ ]] Build a chatbot that will collect patient history information, replacing intake forms and questionnaires patients would otherwise have to complete
****
<div class = "answer">

Three of the five examples are machine learning.

Automatically flagging potentially high-risk patients based on ultrasound images is an example of machine learning.
It would be supervised machine learning, since there is a clear outcome to predict (is the patient high risk?).
The data analyzed would be images; image analysis is an excellent application of machine learning with a lot of active development (see, for example, this [collection in Nature](https://www.nature.com/collections/gfbjhfjfgg)).

Estimating the difference in average number of ER visits per year for patients with and without health insurance is an example of statistics, but not a machine learning model.
It is an example of statistical inference -- measure the difference in number of visits per year for each group in your collected data and take the difference, and use that to estimate (infer) the true difference between those two populations of patients.

Building a model to give the probability that a given patient will be readmitted is another example of supervised machine learning, with a clear outcome variable to predict (readmittance).

Identifying meaningful clusters of patients with similar symptoms would be an example of unsupervised machine learning.

Building a chatbot would be artificial intelligence (AI), but not machine learning per se.

</div>
****

True or False: Big data sets are generally higher quality than smaller data sets.

[( )] TRUE
[(X)] FALSE
****
<div class = "answer">

False.
Working with large datasets does not mean your data are higher quality -- in fact, problems with data quality can be harder to identify in big data, making them that much more problematic.

</div>
****

True or False: If you use data that is nationally representative with respect to things like gender, race, and socioeconomic status, then machine learning models trained on it will not be biased (in the fairness and equity sense of the term).

[( )] TRUE
[(X)] FALSE
****
<div class = "answer">

False.
While using representative data is important for many research questions, it does not guarantee unbiased results.
Bias can come from many different sources, including things like the choice of variables to use (as was the case for [Obermeyer and colleagues, 2019](https://www.science.org/doi/full/10.1126/science.aax2342)) or patterns in the data that reflect structural inequalities in society.

</div>
****

Dr. Bronner has a large dataset of fMRI brain scans, and they want to examine how patterns of activation across the whole brain relate to other aspects of health.
The scan data are very complex, though (more than 100,000 non-zero voxels), and Dr. Bronner is hesitant to estimate a model with that many variables.
What should their first step be?

[( )] prediction
[( )] anomaly detection
[( )] clustering
[(X)] dimension reduction
****
<div class = "answer">

Dimension reduction is often a valuable first step when working with large and complex data, especially in cases where the number of variables (in this example, each voxel would be one variable) is larger than the number of observations (the number of patients scanned).

For an overview of dimension reduction techniques used in neuroimaging analysis, see [Mwangi, Tian and Soares, 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4040248/).

</div>
****

Now Dr. Bronner would like to use their fMRI data along with linked electronic health records (EHR), including diagnoses, to predict whether or not patients have Alzheimer's disease from their brain scans.
What kind of model should they use?

[(X)] prediction
[( )] anomaly detection
[( )] clustering
[( )] dimension reduction
****
<div class = "answer">

This would be a prediction model, specifically a binary classification model.
The training data would be labeled (Alzheimer's diagnosis would be available via the linked EHR), so this is an example of a supervised learning model.

</div>
****

A graduate student in Dr. Bronner's lab wants to analyze the EHR data to identify possible sub-groups of Alzheimer's patients with similar symptoms.
What kind of model should they use?

[( )] prediction
[( )] anomaly detection
[(X)] clustering
[( )] dimension reduction
****
<div class = "answer">

A clustering model would be the best fit here.
The data would be measurements of all of the relevant symptoms (the input variables, or features) for each of the Alzheimer's patients.
There would be no outcome variable, so this is an unsupervised learning model.

</div>
****

## Additional Resources

There are many excellent (and free!) machine learning courses available online.
Here are a few of our favorites:

- [Google's Intro to Machine Learning](https://developers.google.com/machine-learning/intro-to-ml)
- [DeepLearning.ai](https://www.deeplearning.ai/): Several world-class courses on machine learning topics ranging from intro to very advanced
- [Introduction to Statistical Learning at Stanford](https://www.statlearning.com/online-course) (there is also a free [text book](https://hastie.su.domains/ISLR2/ISLRv2_website.pdf) and [R package](https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf) that go with this course)
- [Introduction to Applied Machine Learning at the University of Wisconsin](https://dionysus.psych.wisc.edu/iaml/)

For a fun and detailed introduction to machine learning, check out [Google's comic about AI and machine learning tools](https://cloud.google.com/products/ai/ml-comic-1).
It follows the story of an engineer, Martha, as she learns about machine learning from a cat and a friendly flying robot.

![A flying robot, a cat, and a human software engineer discussing three signs saying "Supervised Learning", "Unsupervised Learning", and "Reinforcement Learning". The robot says, "These categories don’t account for ALL of machine learning, but they cover a lot of ground." The engineer says, "So self-playing arcade games are cool and all —— but I want to hear more about practical applications. What can this stuff be used for in the real world?"](https://cloud.google.com/static/products/ai/ml-comic-1/assets/panel_71_2x.png)

More examples!
---

This module includes links to many examples of machine learning applications, but there are so many more!
Because machine learning covers such a wide range of techniques and models, there are many potential applications in biomedical science.
If you want to read about more examples of machine learning in biomedical research, here's a list to get you started:

- [Predict which tests a patient entering the emergency department might need](https://healthitanalytics.com/news/machine-learning-model-helped-streamline-22-of-pediatric-ed-visits) so they can be ordered automatically and care delivered more quickly
- [More accurately estimate severity of osteoarthritis from knee X-ray images](https://www.nature.com/articles/s41591-020-01192-7), reducing unexplained racial disparities in pain that occur when images are graded by human physicians
- [Assess the feasibility of allocating Medicare funds based on predicted mortality](http://ziadobermeyer.com/wp-content/uploads/2019/09/eolspend.pdf), addressing the question of whether Medicare spending during what turns out to be the last year of life is wasteful from a policy perspective
- [Automatically extract things like symptoms and history from unstructured notes](https://arxiv.org/pdf/2107.02975.pdf), making complicated EHR (electronic health record) data easier to analyze
- [Guide clinicians performing radiofrequency ablation](https://pubmed.ncbi.nlm.nih.gov/30939953/)

## Feedback

@feedback
