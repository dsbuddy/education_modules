<!--

author:   Joy Payton
email:    paytonk@chop.edu
version: 1.5.1
current_version_description: Fixed inaccurate acronym, added links to intro to version control, fixed additional resources structure
module_type: standard
docs_version: 2.0.0
language: en
narrator: US English Female
mode: Textbook
title: Reproducibility, Generalizability, and Reuse
comment:  This module provides learners with an approachable introduction to the concepts and impact of **research reproducibility**, **generalizability**, and **data reuse**, and how technical approaches can help make these goals more attainable.
long_description: **If you currently conduct research or expect to in the future**, the concepts we talk about here are important to grasp.  This material will help you understand much of the current literature and debate around how research should be conducted, and will provide you with a starting point for understanding why some practices (like writing code, even for researchers who have never programmed a computer) are gaining traction in the research field.  **If research doesn't form part of your future plans, but you want to *use* research** (for example, as a clinician or public health official), this material will help you form criteria for what research to consider the most rigorous and useful and help you understand why science can seem to vacillate or be self-contradictory.

estimated_time_in_minutes: 60

@pre_reqs

It is helpful if learners have conducted research, are familiar with -- by reading or writing -- peer-reviewed literature, and have experience using data and methods developed by other people.  There is no need to have any specific scientific or medical domain knowledge or technical background.    
@end

@learning_objectives  

After completion of this module, learners will be able to:

* Explain the benefits of conducting research that is **reproducible** (can be re-done by a different, unaffiliated scientist)
* Describe how technological approaches can help research be more reproducible
* Argue in support of practices that organize and describe documents, datasets, and other files as a way to make research more reproducible

@end


good_first_module: true
collection: intro_to_data_science
coding_required: false

@version_history

Previous versions: 

- [1.4.1](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/2ba39cddbbe6436e18b04ff62f7dfff4406c5880/reproducibility/reproducibility.md#1): Updated quizzes, learner outcomes, highlight boxes
- [1.3.1](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/bfbada6fa70c3c9ef0d027eb2e450990b7c7fac7/reproducibility/reproducibility.md#1): Update template, remove some CHOP-specific references, 
- [1.2.2](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/5f90b59f30dc1f29416df61773d544cf15dce83a/reproducibility/reproducibility.md#1): fix incorrect hyperlink, correct layout and typos

@end 

import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros.md
import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros_r.md
-->

# Reproducibility, Generalizability, and Reuse

@overview

## Before We Get Started

Non-technical tools that you'll need along the way, in this module and elsewhere, include a tolerance for ambiguity, the willingness to be a beginner, and practiced rebuttals to the self-critic / impostor syndrome.

The tools we're describing in this material are often complex and are used differently by different constituencies.  It can be intimidating to learn how to use **git**, for example, from people who work in large software development teams.  Users like these have years of experience with established **agile** project management practices.  They may refer to **milestones**, each related to one or more **sprints**.  They may use short acronyms (**LGTM!**) within **issues**, including linking to **commits**, and have specific naming conventions for each **branch**, along with demands to "always **squash and merge**!".  Teams with established norms may make inside jokes or references that make it seem like any mistake or deviation from procedure is a huge problem.

Do ***you*** have to learn all of this?  What's centrally important, and what's optional or ancillary?  It can be daunting even to simply get started when you run into scenarios like this one.

Jargon abounds in the tech field, and unfortunately, so does **gatekeeping.**

By gatekeeping, we mean that with or without intent, some people with greater technical experience can suggest that people with less experience don't belong or shouldn't participate.  This can take a lot of forms, such as:

* Using lots of TLAs (Three Letter Acronyms) without context or explanation.
* Snarky, unhelpful comments on Stack Overflow when a new user poses a question that doesn't meet the standards of a well-crafted, reproducible example.
* Condescension which can mask insecurity, with words like "well, clearly you should..." or "oh, just...", which add nothing to the conversation except a display of dominance.

We propose an approach that is more gate **opening** than gate **keeping** and includes positive regard such as delight at sharing knowledge.  We're not alone:

<div style = "margin: 2rem auto; max-width: 50%; font-size:0.8em; align: center;">

![XKCD Cartoon, the text of which begins "I try not to make fun of people for admitting they don't know things."  The cartoon depicts sharing knowledge as a fun adventure.](media/ten_thousand_2x.png)

(Image used under a Creative Commons Attribution-NonCommercial 2.5 License.  Original post at https://xkcd.com/1053.)

</div>

<div class = "care">
<b style="color: rgb(var(--color-highlight));">A little encouragement...</b><br>

As learners, we ask that you build your core competencies and non-technical skills by:

* Asking for help with self-education.  Ask colleagues and friends for additional resources as well as criteria for evaluating whether a web page or article you found is useful or not.
* Giving feedback on this and other modules and correcting the authors if we act as gatekeepers.  We were beginners once, too!
* Standing up to your inner critic.  You belong in science, and you deserve access to data science learning.
* Having a sense of humor around failing code.  The people writing these data science modules have to Google syntax, too!  
* Becoming aware of the physical and psychological markers of fatigue, frustration, and the need for a break.
* Recalling your own ability to master difficult material.  The fact that something feels staggeringly difficult today doesn't mean it will always be so challenging.

</div>

### Quiz: Research Realites

True or False: If you're a driven, intelligent researcher, you're unlikely to experience failure as you learn skills like writing code.

[( )] TRUE
[(X)] FALSE
***
<div class = "answer">

FALSE!  Failure is something that people who work a lot with technology have to become comfortable with.  You can even think of the process of writing code as "failing a lot until you get things right, then moving on to failing on the next project."  

Error codes, mistakes, and confusion with new methods can be frustrating, especially if you have a lot of confidence and competence in your current way of working.  It's easier said than done, but you might find it helpful to recall that failure is a critically important tool in science, even if it's a tool we don't love to talk about.

</div>
***

## Concepts

The concepts of **reproducibility**, **generalizability**, and **reuse** will frame the problem space that we'll describe in this module.  We'll define these terms and give some examples.  

These concepts will be illustrated in a (charming if rage-inducing) YouTube video, *A Data Management and Sharing Snafu*.

After exploring these concepts, we'll go over some methods to address these challenges, using technology.

### Reproducibility

You may hear the terms **"reproducibility"** and/or **"replicability"**, depending on context.  Jargon varies by field and you may see either or both terms used to refer to similar goals: the ability to (1) precisely redo analyses on original data to check the original findings or (2) to carefully apply the original methods to new data to test findings in a different dataset.  Here, we usually follow what is becoming more customary and use the term "reproducible" to refer to both efforts.

The **"reproducibility crisis"** refers to the problem in peer-reviewed research in which studies *cannot* be reproduced or replicated because of insufficient information, or in which studies *fail* to be reproduced or replicated because of preventable problems in the initial research.  This is problematic because it means wasted time and money, reduced public trust in science, unverifiable claims, and lost chances for scientific consensus.

-------------

<div style = "clear:both;">

<div style = "max-width: 45%; float:left;"> 

If you've ever tried to reproduce an analysis or study procedure from just the methods section of a manuscript, you probably experienced it as something like "drawing a horse" as shown here.

Providing vague methods that can't be easily reproduced can be a product of many factors influencing manuscript authors, such as:

* Preserving word count for other sections
* Assuming that implicit steps will be understood by others
* Feeling vulnerable about close scrutiny
* Not having methods well documented to begin with
* Obscuring practices that might prompt manuscript rejection or unfavorable review

When describing a multi-step task that should be able to be carried out by others, explicit step by step instructions are key.

</div>

<div style = "margin-left: 2em; margin-top: 3em; max-width: 45%; float:left;">
![Humorous illustration of "how to draw a horse" which shows trivial complexity until the final step, "add small details".  This last step is not explained but provides most of the drawing of the horse.](media/horse.png)

Courtesy of artist. [Original work](https://oktop.tumblr.com/post/15352780846)

</div>

</div>

<div style = "clear:both;"></div>

--------------

Examples of reproducibility problems exist at small and large scale.  Importantly, reproducibility affects not just interaction between scientific collaborators (or rivals), but also between "current you" and "you six months ago".  Perhaps you have felt the impact of non-reproducible research:

* Experiencing dread at trying to reproduce your own findings a few months after doing it the first time
* Being stymied by a collaborator's cryptic notes that don't explain how to do a particular analysis step
* Being unable to perform required computation because of reliance on expensive, deprecated, or proprietary software or hardware
* Results that don't replicate due to poor statistical practices, such as "p-hacking", "HARKing", convenient outlier selection, or multiple tests without correction

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

Think about it: when have you been frustrated by a process or study that had poor reproducibility?  Have you ever put **yourself** in a bad situation because you didn't think ahead to how you'd need to replicate your actions?

</div>

#### Quiz: Reproducibility

Who benefits from reproducible research practices?  Choose all that apply.

[[X]] The original authors of novel research
[[X]] Patient populations
[[X]] Journal editors and peer reviewers
[[X]] Taxpayers
[[X]] Authors of meta-analyses
[[?]] There are multiple correct answers!
***
<div class = "answer">

All of these groups benefit!  

Researchers who publish novel studies that can be reproduced will benefit from having more of their peers use their methods, data, and statistical approaches, which means more citations and greater influence.  Researchers may also be able to get their manuscripts into higher reputation journals than if their research was not reproducible.

Patient populations benefit from evidence based research that is robust and demonstrated in multiple settings by various teams.  Reproducible research is a key part of translating findings to clinical care.

Journal editors and peer reviewers benefit when submitters use reproducible methods because they can more quickly assess the quality of the research, test the statistical assumptions, and ensure that work can be tested by future investigation.  

Taxpayers benefit when government funded research has the greatest generalizability and highest quality.  Reproducible methods allow a single funded study to have a ripple effect that will continue to influence scientific knowledge well into the future.

Authors of meta-analyses benefit from reproducible practices like data and script sharing, because it allows them to check the findings asserted within a manuscript, compare its findings to those of other manuscripts, and discover differences between analyses that may point to the best methods to use in the practice of science in a given area.

</div>
***

### Generalizability

Research is **generalizable** if findings can be applied to a broad population.  

Historically, biomedical and social science research projects have struggled with generalizability due to unrepresentative data.  For example, the acronym **"WEIRD"** refers to the tendency of psychological studies to rely on subjects (often undergraduate students) who are disproportionately from **W**estern, **E**ducated, **I**ndustrialized, **R**ich, and **D**emocratic cultures -- cultures which, compared with the global population as a whole, are indeed weird.  


<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

Read more: in 2010 Joseph Henrich and others published a brief in *Nature* coining "WEIRD" to describe skewed participation in psychological studies.
<div style = "margin-left: 40px; text-indent: -40px; font-size:0.8em;">

Henrich, J., Heine, S. & Norenzayan, A. Most people are not WEIRD. Nature 466, 29 (2010). [https://doi.org/10.1038/466029a](https://doi.org/10.1038/466029a).

</div>

</div>

Until recently, many biomedical studies were conducted on disproportionately male populations and ignored disease presentation, physiology, and pharmacodynamics in women and girls (or even female lab animals).  In 1993, the [NIH Revitalization Act](https://www.ncbi.nlm.nih.gov/books/NBK236531/) began requiring NIH-funded clinical research to include women as subjects.  

This mandate did not require the same inclusivity in bench research, but NIH encouraged adoption of sex-balanced research outside of human subjects.  Two decades after the 1993 legislation, Janine Clayton and Francis Collins wrote a [pointed call to action in *Nature*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5101948/), indicating that bench researchers had not willingly followed best practices and that NIH apparently needed to require the use of female animals and cells:

> There has not been a corresponding revolution in experimental design and analyses in cell and animal research — despite multiple calls to action. Publications often continue to neglect sex-based considerations and analyses in preclinical studies. Reviewers, for the most part, are not attuned to this failure. The over-reliance on male animals and cells in preclinical research obscures key sex differences that could guide clinical studies. And it might be harmful: women experience higher rates of adverse drug reactions than men do. Furthermore, inadequate inclusion of female cells and animals in experiments and inadequate analysis of data by sex may well contribute to the troubling rise of irreproducibility in preclinical biomedical research, which the NIH is now actively working to address.

In early 2016, a policy requiring the consideration of sex as a biological variable (SABV) went into effect, and applications for NIH funding were required to comply with best practices related to sex-inclusive experimental design.  Progress in NIH's SABV efforts were [recently reported in a 2020 article](https://pubmed.ncbi.nlm.nih.gov/31971851/).

<div class = "history">
<b style="color: rgb(var(--color-highlight));">Historical context</b><br>

[Listen to Janine Clayton speak about scientific rigor and female animal inclusion (5 minute listen)](https://www.wbur.org/hereandnow/2014/05/20/nih-female-animals).  A partial transcript accompanies the audio.

</div>

Human bias doesn't just lead to potentially misleading studies, but to potentially misleading research tools as well.  For example, in wearable sensor and computer vision development, engineers using skewed samples failed to realize that the optical sensors and computer vision algorithms they created may perform less well on dark skin. See, for example, [a *STAT* piece about Fitbits](https://www.statnews.com/2019/07/24/fitbit-accuracy-dark-skin/) and [a *Boston Magazine* report about MIT research on AI and racial bias](https://www.bostonmagazine.com/news/2018/02/23/artificial-intelligence-race-dark-skin-bias/).

The challenge of generalizability is closely linked to reproducibility.  For example, a study that demonstrates the effectiveness of exercise to improve functioning in depressed suburban teenagers may not generalize to city-dwelling adults.  In order to gain broader generalizability, this promising experiment on a limited population should be reproduced in a broader or different population.  If the original study is difficult to reproduce, however, such broader application may prove impossible.

Technological solutions alone cannot correct human problems such as recruitment bias or white overrepresentation in research personnel.  Careful use of technology can, however, add to research transparency and reproducibility and promote honest disclosure of challenges to generalizability.

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

Consider asking yourself:

* Can research bias be quantified and disclosed using technology?
* How can bias be reduced and generalizability improved in your research area?

</div>


### Reuse

In addition to reproducibility, another important element of research is the ability to **reuse** assets such as data and methods to related research that may not be a direct replication.  

For example: 

* Researchers may hypothesize that a computer vision approach used to analyze moles might be useful as well in other areas of medicine that need edge detection, such as tumor classification.
* Longitudinal data that provides rich phenotyping of a cohort of patients with hypermobility syndromes may be useful not just to the original orthopedic researcher community but also to cardiologists interested in comorbid vascular and autonomic nervous system conditions.  

The reuse of research data and methods allows researchers to collaborate in ways that advance cross-domain knowledge and professional interaction, as well as honoring the time and energy of human subjects whose data can be leveraged to get as much scientific value as possible.

The reuse of data and other research assets has numerous challenges.  You may have experienced problems in this area such as:

* Encountering resistance to sharing data, methods, scripts, or other artifacts
* Data that is not well described or labeled, or is stored in a "supplemental materials" page without context
* Overly strict informed consent documents that prevent researchers from reusing their own data or sharing it with colleagues

Planning ahead for data reuse is an important part of grant writing, experimental design, interactions with Ethics Committees and Institutional Review Boards, documentation of data and methods, and much more!

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

Want to get a quick overview of some of the privacy practices that regulate responsible research data sharing in biomedicine?  Check out [a brief article with an overview of privacy practices related to data sharing in research](https://education.arcus.chop.edu/privacy-overview/).

</div>

### A Data Management and Sharing Snafu

This is an approachable and humorous introduction to the practical impact of poor research practices leading to downstream impact.  

As you listen to the video, try to identify problematic research practices which could have been prevented by more careful use of technology.  Which of these mistakes have you encountered personally?  Which have you committed?

!?[A Data Management and Sharing Snafu, which depicts an interaction between two scientists, drawn as cartoon animals](https://www.youtube.com/watch?v=66oNv_DJuPc?cc_load_policy=1)

#### Quiz: Dr. Judy Benign

In the next section of this module, we'll introduce some technical solutions.  Let's start with a quick pre-check of your familiarity with technology and how it can help prevent research problems.

Which of these problems are examples of the kinds of problems that may have a potential technological fix?

[[ ]] Unwillingness to share data
[[X]] Lack of clarity about what variable names mean
[[X]] Not remembering where data is located
[[X]] Software becoming unavailable
[[ ]] Mentors relying on postdocs to do most of the work
[[?]] Hint: We consider three of these to be problems with potential technological fixes!
***
<div class = "answer">

While technology alone can't motivate researchers to change some behavior, like being skeptical about data sharing in general or turfing much of the hard work of analysis to junior researchers, technology can help save us from ourselves in other ways.

For example, it's understandable that short variable names may be hard to connect to their full meaning, and the creation of a **data dictionary** might have helped avoid the problem of trying to decode the meaning of "SAM1" and "SAM2".  

Researchers are busy, lab churn is a fact of life, and staff like research assistants and postdocs can move on in the middle of a project.  That's why consistently applying **data management** best practices such as shared drives, version controlled repositories, or automated backup can be helpful in preventing misplaced files.  

The careful listing of **dependencies** like software can help quantify the risk of data becoming unusable, and data management practices can include saving plain text versions of encoded data, so that if a proprietary data format is no longer easily usable, a version of the data exists that can still have some utility.

</div>
***

## Tools for Better Practices

In the "Data Management and Sharing Snafu" video, we hear some common researcher errors that are hard to prevent with technology, such as a tendency to gloss over questions with a bit of arrogance: "Everything you need to know is in the article...".  However, technology would have helped solve some of the problems that Dr. Judy Benign had to deal with.  

Here we will provide a broad overview of how some tools and practices (scripts, data management and metadata, version control, and dependency management) can ameliorate some of the challenges we've outlined earlier.  Technology alone cannot solve the reproducibility crisis, but tools can support researchers who are trying to apply rigor and clarity to their research efforts.

Areas we won't cover here, but are critical to the consistent production of reproducible science, include researcher bias, research incentivization, publication bias, research culture, mentorship, and more.  While we assert that proper use of technology is a **necessary** part of reproducible science, technology alone is not **sufficient**.

### Scripts

**Scripts**, in this context, are a series of computer code instructions that handle elements of research such as:

* Ingesting data (for example, accessing a .csv file or downloading the latest data from REDCap)
* Reshaping and cleaning data (such as removing rows that don't meet given conditions for completeness or correctness, combining data from two or more sources, or creating a new field using two or more existing fields)
* Reporting statistical characteristics (for example, finding quartiles, median values, or standard deviations)
* Conducting statistical tests (e.g. ANOVA, two-sample t-tests, Cohen's effect size)
* Creating models (such as a linear or logistic model or a more complex machine learning algorithm like clustering or random forest classification)
* Saving interim datasets (e.g. storing a "cleaned" version of data for use in later steps, or creating a deidentified version of data)
* Creating data visualizations (such as boxplots, Q-Q plots, ROCs, and many more)
* Communicating methods and findings in a step-by-step way (e.g. writing a methods section from within the steps of analysis)
* And more....

Scripts may be written in free, open source tools like R, Python, Julia, and Octave, or, with care, can be extracted from commercial tools (for instance, by using a syntax file).  It's important to realize that good scripts are complete and don't rely on human memory of steps that aren't recorded in the script.  For example, using a point-and-click solution like Excel to clean data prior to analyzing it using code relies on human memory of what was done in Excel.

<div class = "behind-the-scenes">
<b style="color: rgb(var(--color-highlight));">Behind the scenes</b><br>

**What about scripting options in Excel, SAS, SPSS, and Stata?**

Excel (and some other primarily point-and-click software) *can* support scripted analysis.
Tools like PowerQuery and PowerBI make it possible for you to write out data cleaning steps as a script in Excel, which is a huge advantage over doing those steps interactively by clicking menus and typing in cells. 

You may also have been exposed to SAS, SPSS, and Stata, all of which have a point-and-click element as well as the possibility of scripted analysis. 

If you can switch from relying on point-and-click procedures to exclusively using the scripting options in your analysis software, that will make your work much more reproducible.
Keep in mind, though, that if **any** steps at all in your process rely on something done outside of the script, that greatly undermines the reproducibility of your analysis.

</div>

Although tools like Excel *can* use scripts, many users of these tools depend on un-scripted actions such as cleaning data beforehand in a separate program and the use of point-and-click, menu driven selections.  
We can contrast that with tools like R and Python that are built for scripting first, where every step is explicitly recorded. 

For this reason, we suggest the use of R and Python for most research purposes, along with knowledge of how to use the data querying language SQL (pronounced "sequel").  

Advantages of these tools include:

* R, Python, and SQL are widely used, well-documented and tested.
* R and Python especially have a scientific and medical user base that is friendly for beginners.  
* These tools are free and open-source, which allows for greater reproducibility, including in lower-resourced settings.  

However, learning R, Python, SQL, and other skills requires an investment of time and energy that can be difficult for a busy clinician or scientist to justify, especially when one has already developed considerable experience in point-and-click analysis.

It's worth considering the words of an archaeological team that wrote an article about reproducible research for a lay audience in a [2017 *Slate* article](https://slate.com/technology/2017/07/how-to-make-a-study-reproducible.html):

<div>

>However, while many researchers do this work by pointing and clicking using off-the-shelf software, we tried as much as possible to write scripts in the R programming language.<br/><br/>Pointing and clicking generally leaves no traces of important decisions made during data analysis. Mouse-driven analyses leave the researcher with a final result, but none of the steps to get that result is saved. This makes it difficult to retrace the steps of an analysis, and check the assumptions made by the researcher. <br/><br/> ... <br/><br/>It's easy to understand why many researchers prefer point-and-click over writing scripts for their data analysis. Often that's what they were taught as students. It's hard work and time-consuming to learn new analysis tools among the pressures of teaching, applying for grants, doing fieldwork and writing publications. Despite these challenges, there is an accelerating shift away from point-and-click toward scripted analyses in many areas of science.

</div>

### Data management and metadata

**Data management** includes the organization, annotation, and preservation of data and metadata related to your research or clinical project.

Data management is a critical pain point for many data users.  What's the best way to wrangle the data files needed to carry out a project?  Should documents be stored in a common drive?  Using what kinds of subfolders?  How should researchers deal with emails that are sent back and forth between researchers to define a specific cohort?  What is the long-term storage strategy for this data?  Are there ways to save the data in multiple formats to accomodate unknown future needs?  

Even a small project organized by a single researcher can be complex, and when a team of several researchers and supporting staff are involved, individual data management practices can collide.  A few topics that fall under the category of data management include:

* File naming standards for project files
* The format in which data is collected, and where it is stored
* How data is backed up and kept private
* Where regulatory files such as protocols are kept
* How processes and procedures are stored and kept up to date
* Who has access to what assets and when that access expires

Importantly, NIH has begun requiring more rigorous data sharing & management plans for all grants.  This requirement went into effect in January 2023.  Data management planning is now a critical skill for grant writing.

**Metadata** is, in its simplest definition, data about data. Some examples of metadata might include:

* Who collected the data?
* When was the data collected?
* What units does the data use?
* What kind of thing is the data measuring?
* What are the expected values?
* Are there any codes for specific cases (e.g. missing vs. unwilling to answer vs. does not apply)?

Metadata can be found in many places. Sometimes it's implicit, as when it appears in variable names.  The variable "weight_kg", for example, discloses both the measure and the units. Often metadata is found more fully explained in **data dictionaries** or **codebooks**, where variables in a dataset are described more completely. Sometimes metadata can be found almost in passing, mentioned in an abstract or in the methods section of a paper, or in some descriptive text that accompanies a data download or a figure.

Creating useful metadata is a crucial step in reproducible science.  It's essential in helping run an efficient project involving multiple people, since helpful metadata can help reduce incorrect data collection, recording, and storage.  Metadata can help explain or contextualize some findings (e.g. when the time of day of a blood draw affects lab results).  It can also support the use, discovery, and access of data over time.

Metadata can exist at various levels of a project. For example, some metadata is overarching and describes an entire project (e.g. the institution that oversaw the data collection), while other metadata adheres to a specific data field (e.g. the make and model of a medical device that gave a certain measurement).  

REDCap is one example of software that explicitly creates a data dictionary that includes information such as the name and description of a variable, the kind of input that can appear there (alphanumeric, numeric, email format, etc.), and whether a field could identify a subject or not.  

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

These [Research Data Management Resources](https://www.research.chop.edu/applications/arcus/resources) tend to be very practical and can be used right away to improve data management!  Simply click on the link and then expand the part of the accordion labeled "Research Data Management Resources (Public Access)".

</div>

#### Quiz: Data Management and Metadata

Which of the following is true about data organization for researchers?

[[X]] Data management planning is a critical part of grant writing, research planning, and the conduct of research.
[[ ]] Metadata is specialized information used in complex computational projects and is best created by expert information scientists
[[X]] When working with research data, small decisions, like column naming conventions, can have big implications for research success.
[[ ]] Major funders of biomedical research understand that data management techniques are easy to carry out, and don't need these methods to be described.
[[?]] Hint: There are two correct answers.
***
<div class = "answer">

Data management planning is indeed a critical part of research, including grant writing.  In fact, NIH requires careful documentation of data management planning for new research (as of 2023).  

Data management is both mundane (it includes seemingly simple decisions such as what to name a directory of files) and high-stakes (poor data management can make research slow down or even fail).  The fact that scientists constantly make incremental data management decisions, usually without consideration of downstream effects means that this skill is complex and requires practice and explicit planning.  It's not easy!

Metadata (data about research data) doesn't have to be highly technical and specialized.  Metadata is present in every research project and includes simple information such as who collected the data and in what location.  The careful recording of metadata is important for research success and reproducibility.  For example, as we discussed earlier in our discussion of [A Data Management and Sharing Snafu](https://www.youtube.com/watch?v=66oNv_DJuPc?cc_load_policy=1), the creation of metadata, like a **data dictionary** might have helped researchers avoid the problem of trying to decode the meaning of "SAM1" and "SAM2". 

</div>
***

### Version Control

Version control is the discipline of tracking changes to files in a way that captures important information such as who made the change and why.  It also allows for reversion to previous versions of a file.

Many of us use "home grown" version control systems, such as using file names to capture who most recently added comments to a grant proposal or the date of a particular data download.  The difficulty with this is that each member of a team or lab may use different file naming protocols, and within a few months, the number of files can proliferate wildly.  Collaborators may feel unsure about deleting old versions of files, and data and file hoarding leads to delays and confusion.

Technological solutions for version control have been around for decades, and one system in particular has won the bulk of the market share in version control -- **git**.  Git is free, open source version control software.  

We won't go into details at this point, but one of the helpful aspects of git is that it allows you to see what changed, when, and by whom, in your files, along with a helpful (we hope) comment describing the change.  See below for a humorous interpretation (which isn't that far off the mark).

<div style = "margin: 2rem auto; max-width: 50%; font-size:0.8em; align: center;">

![XKCD Cartoon showing a git commit history that devolves into messages like "my hands are just typing words".  Captioned "As a project drags on, my git commit messages get less and less informative."](media/git_commit_2x.png)

(Image used under a Creative Commons Attribution-NonCommercial 2.5 License.  Original post at https://xkcd.com/1296 .)

</div>

Git and GitHub are distinct organizations with different products.  Git is a free, open-source version control system, and GitHub is a company that provides services to make it easier to use git for software development and related uses.  GitHub has free tier use as well as paid services.  

Some institutions pay for an enterprise version of GitHub that may be accessible only to institutional users on a secure network, and not available to the general public.  For science that can be more broadly shared, many researchers use the publicly available [GitHub.com](https://github.com), which is a website run by GitHub.  As an example of another GitHub resource, many git users find that the [GitHub Desktop](https://desktop.github.com/) software is useful for working with git on their local computers.

<div class = "learn-more">
<b style="color: rgb(var(--color-highlight));">Learning connection</b><br>

To learn more about this topic, see our [Intro to Version Control module](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/git_intro/git_intro.md#1).

</div>

### Dependency Management

If you've ever created a slide show in one computer only to have it look terrible in another, you know the problem that **dependencies** can cause.  Dependencies are technical requirements that, if not met, can cause problems.

In our slide show example, dependencies could include:

* Certain fonts
* Display settings
* PowerPoint version
* Access to image or sound files needed in the slide show 

Dependencies that are well-documented and understood will help make research more reproducible and help make problems more fixable.  Dependencies that are undocumented or not known about will inevitably cause problems that are frustrating to troubleshoot.

Sometimes it isn't clear whether something is really a dependency (this value or program *must* be the same as what you used) or just a circumstance (you used a particular version of Python but there's no reason to think that previous or subsequent versions wouldn't work just as well).  For this reason, recording both known and possible dependencies is a helpful practice. Common dependencies in research and data analytics include:

* Operating system: does your use of particular software require the use of Microsoft Windows 10 or later?
* Regional data formatting: does your analysis assume that decimal values use a period, not a comma, to set off the decimal value?
* Program versions: did you conduct your analysis in R 3.7?  Have you tried it in 4.1?
* Technical data formatting: does your analysis expect a .csv of data with specified columns holding certain measures?
* Access to reference files: are you aligning to hg38 or to a previous reference genome?
* Hardware requirements: does your research paradigm require a particular kind of hardware for generating stimuli or recording response?

Dependency management is an approach that makes it easier to determine the precise set of tools and inputs required by your data collection and analysis.  Every research effort should document which tools were used and which version of each was employed.  This can be as simple as a text file, or could include installation instructions or even a file that includes the exact versions of software used in the original research, in order to create a computer environment that can perform the analysis under the original conditions.

## Additional Resources

Enjoy some supplemental materials that you might find useful, and feel free to suggest additions!  

<h3>Center for Open Science</h3>
[The Center for Open Science (COS)](https://www.cos.io/services/training) is one of the foremost thought leaders in reproducible science. 
COS has built out a collection of materials on a variety of subjects including data management, how to write reproducible methods, and sharing your research which can be accessed on the [COS Training](https://www.cos.io/services/training) page.

<h3>John Oliver on Reproducibility</h3>
[A 20-minute long (intermittently not safe for work) clip from John Oliver](https://www.youtube.com/watch?v=0Rnq1NpHdmw)'s infotainment show "Last Week Tonight" includes off-color language and adult references but is a great introduction to topics including:

* p-hacking
* non-publication of null findings (the "file drawer" problem)
* replication studies
* incentivization problems in research
* scientific communication in mass media
* generalizability
* public support of science
* industry funding

<h3>For Excel Users</h3>

* An educator [shares some harm reduction techniques for Excel users](https://education.arcus.chop.edu/excel-caveats/)
* A former user of Excel [shares why he's moved on to using scripted code and gives helpful hints to those still using Excel](https://education.arcus.chop.edu/the-spreadsheet-betrayal/)

<h3>Academic resources</h3>

For a list of excellent readings on reproducibility in psychology, see the [mock syllabus for a class called "PSY 607: Everything is F*cked"](https://thehardestscience.com/2016/08/11/everything-is-fucked-the-syllabus/) posted by Prof. Sanjay Srivastava (note that the syllabus was posted as a joke and includes off-color language, but the readings listed are very real and provide a great introduction to reproducibility in science). 

## Feedback
@feedback