import pandas as pd
df=pd.DataFrame()

df.loc["bash_103_combining_commands", "title"] = "Bash: Combining Commands"
df.loc["bash_103_combining_commands", "author"] = "Elizabeth Drellich and Nicole Feldman"
df.loc["bash_103_combining_commands", "estimated_time_in_minutes"] = ""
df.loc["bash_103_combining_commands", "comment"] = "This module will teach you how to combine two or more commands in bash to create more complicated pipelines in Bash." 
df.loc["bash_103_combining_commands", "long_description"] = "This module is for learners who can use some basic Bash commands and want to learn to how to use the output of one command as the input for another command." 
df.loc["bash_103_combining_commands", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Use the commands `wc`, `head`, `tail`,`sort`, and `uniq`&- Redirect output to a file using `>` and `>>`&- Chain commands directly using the pipe `|`&" 
df.loc["bash_command_line_101", "title"] = "Bash / Command Line 101"
df.loc["bash_command_line_101", "author"] = "Nicole Feldman and Elizabeth Drellich"
df.loc["bash_command_line_101", "estimated_time_in_minutes"] = ""
df.loc["bash_command_line_101", "comment"] = "This course teaches learners to navigate their computer, as well as view and edit files, from the command line using Bash." 
df.loc["bash_command_line_101", "long_description"] = "This course is designed to be both an introduction to bash / command line for those who are total newbies as well as refresher for those some with experience running code who want a more solid command of the basics." 
df.loc["bash_command_line_101", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Describe what bash scripting is and why they might want to learn it for data management and research&- Navigate their file system using the bash shell&- View and edit the contents of a file from the bash shell&" 
df.loc["bash_command_line_102", "title"] = "Bash: Searching and Organizing Files"
df.loc["bash_command_line_102", "author"] = "Nicole Feldman and Elizabeth Drellich"
df.loc["bash_command_line_102", "estimated_time_in_minutes"] = "30"
df.loc["bash_command_line_102", "good_first_module"] = "false" 
df.loc["bash_command_line_102", "coding_required"] = "true"
df.loc["bash_command_line_102", "coding_language"] = "bash"
df.loc["bash_command_line_102", "coding_level"] = "basic"
df.loc["bash_command_line_102", "sequence_name"] = "bash_basics"
df.loc["bash_command_line_102", "data_task"] = "data_management"
df.loc["bash_command_line_102", "comment"] = "This module will teach you how to use the bash shell to search and organize your files." 
df.loc["bash_command_line_102", "long_description"] = "This module is for people who have a bit of experience with bash scripting and want to learn to use its power to organize their file and folders." 
df.loc["bash_command_line_102", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Search existing files for particular character strings.&- Search folders for files with certain titles.&- Move files to new locations in a directory system.&- Copy files and directories.&- Delete files and directories.&&" 
df.loc["bash_command_line_102", "Prerequisties"] = "&Learners should be familiar with using a bash shell to navigate a directory system. Learners will get the most out of this lesson if they can also create directories and files, write text to files, and read files from their bash shell command line interface.&&" 
df.loc["bash_command_line_102", "Sets You Up For"] = "   bash_103_combining_commands   bash_conditionals_loops  " 
df.loc["bash_command_line_102", "Depends On Knowledge In"] = "  bash_command_line_101  " 
df.loc["bash_conditionals_loops", "title"] = "Bash: Conditionals and Loops"
df.loc["bash_conditionals_loops", "author"] = "Elizabeth Drellich"
df.loc["bash_conditionals_loops", "estimated_time_in_minutes"] = "60"
df.loc["bash_conditionals_loops", "good_first_module"] = "false" 
df.loc["bash_conditionals_loops", "coding_required"] = "true"
df.loc["bash_conditionals_loops", "coding_language"] = "bash"
df.loc["bash_conditionals_loops", "coding_level"] = "intermediate"
df.loc["bash_conditionals_loops", "sequence_name"] = "bash_basics"
df.loc["bash_conditionals_loops", "comment"] = "This module teaches you how to iterate through +for+ loops and write conditional statements in Bash." 
df.loc["bash_conditionals_loops", "long_description"] = "This lesson teaches the basics of loops (for all x, do y) and conditional statements (if x is true, do y) in Bash. Since the grammar of Bash can be non-intuitive this module is appropriate both for learners who have experience with conditionals and loops in other languages, as well as learners who are learning about these kinds of commands for the first time." 
df.loc["bash_conditionals_loops", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Understand how a +for loop+ works&- Write a +for loop+ in Bash  &- Understand how an +if/then+ statement works&- Recognize and reuse +if/then+ statements in Bash&&" 
df.loc["bash_conditionals_loops", "Prerequisties"] = "Only basic exposure to Bash is expected. The following is a list of actions and commands that will be used without explanation in this module. Each includes a link to help you brush up on the commands or learn them for the first time.&&* [Navigating](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_101/bash_command_line_101.md) a filesystem from a command line interface&* Reading the contents of files with [`cat`](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_101/bash_command_line_101.md#15)&* Writing text to files with [`echo` and `>>`](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_101/bash_command_line_101.md#14)&* Matching character strings with the [character wildcard `*`](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_102/bash_command_line_102.md#4)&" 
df.loc["bash_conditionals_loops", "Sets You Up For"] = "   bash_scripts  " 
df.loc["bash_conditionals_loops", "Depends On Knowledge In"] = "   bash_command_line_101   bash_command_line_102  " 
df.loc["bash_scripts", "title"] = "Bash: Reusable Scripts"
df.loc["bash_scripts", "author"] = "Elizabeth Drellich"
df.loc["bash_scripts", "estimated_time_in_minutes"] = ""
df.loc["bash_scripts", "comment"] = "This module will teach you how to create and use simple Bash scripts to make repetitive tasks as simple as possible. " 
df.loc["bash_scripts", "long_description"] = "If you have some experience with Bash and want to learn how to save and reuse Bash processes, this lesson will teach you how to write your own Bash scripts and understand and use simple scripts written by others." 
df.loc["bash_scripts", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Identify the structure of a Bash script&- Run existing Bash scripts&- Write simple Bash scripts&" 
df.loc["bias_variance_tradeoff", "title"] = "Understanding the Bias-Variance Tradeoff"
df.loc["bias_variance_tradeoff", "author"] = "Rose Hartman"
df.loc["bias_variance_tradeoff", "estimated_time_in_minutes"] = "20"
df.loc["bias_variance_tradeoff", "good_first_module"] = "false" 
df.loc["bias_variance_tradeoff", "comment"] = "The bias-variance tradeoff is a central issue in nearly all machine learning analyses. This module explains what the tradeoff is, why it matters for machine learning, and what you can do to manage it in your own analyses. " 
df.loc["bias_variance_tradeoff", "long_description"] = "Whether you're new to machine learning or just looking to deepen your knowledge, this module will provide the background to help you understand machine learning models better. This is a conceptual module only; there will be no hands-on exercises, so no coding experience is required. " 
df.loc["bias_variance_tradeoff", "learning_objectives"] = "After completion of this module, learners will be able to:&&- define bias and variance as they apply to machine learning&- explain the bias-variance tradeoff&- recognize techniques designed to manage the bias-variance tradeoff&&" 
df.loc["bias_variance_tradeoff", "Prerequisties"] = "&This module assumes learners have been exposed to introductory statistics, like the distinction between [continuous and discrete variables](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-and-continuous-random-variables), [linear and quadratic relationships](https://www.khanacademy.org/math/statistics-probability/advanced-regression-inference-transforming#nonlinear-regression), and [ordinary least squares regression](https://www.youtube.com/watch?v=nk2CQITm_eo).&It's fine if you don't know how to conduct a regression analysis, but you should be familiar with the concept.&&" 
df.loc["bias_variance_tradeoff", "Sets You Up For"] = " " 
df.loc["bias_variance_tradeoff", "Depends On Knowledge In"] = "   demystifying_machine_learning  " 
df.loc["citizen_science", "title"] = "Citizen Science"
df.loc["citizen_science", "author"] = "Rose Hartman"
df.loc["citizen_science", "estimated_time_in_minutes"] = "45"
df.loc["citizen_science", "good_first_module"] = "false" 
df.loc["citizen_science", "comment"] = "This is an overview of citizen science for biomedical researchers." 
df.loc["citizen_science", "long_description"] = "This module covers the what, who, why, and how of citizen science research: what citizen science is, who volunteers, why citizen science might be a good choice for your research, and options for how to get started. Throughout, it highlights several examples of real citizen science projects being used in biomedical research and related fields. No prior knowledge is assumed." 
df.loc["citizen_science", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- list several ways members of the public can contribute to scientific projects&- recognize several different factors that motivate people to volunteer in citizen science&- identify research questions that may be a particularly good fit for citizen science&- examine published materials from citizen science projects for things like policies on collaboration and strategies for implementation&&" 
df.loc["citizen_science", "Prerequisties"] = "None.&" 
df.loc["citizen_science", "Sets You Up For"] = " " 
df.loc["citizen_science", "Depends On Knowledge In"] = " " 
df.loc["data_management_basics", "title"] = "Research Data Management Basics"
df.loc["data_management_basics", "author"] = "Ene Belleh"
df.loc["data_management_basics", "estimated_time_in_minutes"] = ""
df.loc["data_management_basics", "comment"] = "Learn the basics about research data management." 
df.loc["data_management_basics", "long_description"] = "If you conduct research or work with research data or researchers, it's likely that research data management topics affect you.  Learn what research data management is, how to think about it in a structured way, and understand its scientific importance." 
df.loc["data_management_basics", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define research data management&- Explain why data management forms an important part of the responsible conduct of research&- Explain how various research stakeholders share responsibility for research data management&- Give examples of research data management tasks within various stages of the research lifecycle&&" 
df.loc["data_storage_models", "title"] = "Types of Data Storage Solutions"
df.loc["data_storage_models", "author"] = "Nicole Feldman"
df.loc["data_storage_models", "estimated_time_in_minutes"] = ""
df.loc["data_storage_models", "comment"] = "This course will focus on different data storage solutions available to an end user and the unique characteristics of each type. This course will also cover how each storage type impacts one's access to data and computing capabilities." 
df.loc["data_storage_models", "long_description"] = "This module is for people interested in understanding the types of data storage solutions available to them at their institution and why they might want to store their files or perform certain computational tasks in each." 
df.loc["data_storage_models", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Identify and describe different data storage solutions&- Understand the benefits and the limitations of each covered storage solution&- Describe what computational tasks are best suited to the described data storage types&- Know the upfront costs and ongoing maintenance the various storage solutions require&&" 
df.loc["data_visualization_in_ggplot2", "title"] = "Data Visualization in ggplot2"
df.loc["data_visualization_in_ggplot2", "author"] = "Rose Hartman"
df.loc["data_visualization_in_ggplot2", "estimated_time_in_minutes"] = "60"
df.loc["data_visualization_in_ggplot2", "good_first_module"] = "false" 
df.loc["data_visualization_in_ggplot2", "coding_required"] = "true"
df.loc["data_visualization_in_ggplot2", "coding_language"] = "r"
df.loc["data_visualization_in_ggplot2", "coding_level"] = "basic"
df.loc["data_visualization_in_ggplot2", "sequence_name"] = "data_visualization"
df.loc["data_visualization_in_ggplot2", "data_task"] = "data_visualization"
df.loc["data_visualization_in_ggplot2", "comment"] = "This module includes code and explanations for several popular data visualizations, using R's ggplot2 package. It also includes examples of how to modify ggplot2 plots to customize them for different uses (e.g. adhering to journal requirements for visualizations)." 
df.loc["data_visualization_in_ggplot2", "long_description"] = "You can use the ggplot2 library in R to make many different kinds of data visualizations (also called plots, or charts), including scatterplots, histograms, line plots, and trend lines. This module provides an example of each of these kinds of plots, including R code to make them using the ggplot2 library. It may be hard to follow if you are brand new to R, but it is appropriate for beginners with at least a small amount of R experience." 
df.loc["data_visualization_in_ggplot2", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- use ggplot2 to create several common data visualizations&- customize some elements of a plot, and know where to look to learn how to customize others&&" 
df.loc["data_visualization_in_ggplot2", "Prerequisties"] = "&This module assumes some familiarity with principles of data visualizations as applied in the ggplot2 library. If you've used ggplot2 (or python's seaborn) a little already and are just looking to extend your skills, this module should be right for you. If you are brand new to ggplot2 and seaborn, start with the overview of [data visualizations in open source software](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/data_visualization_in_open_source_software/data_visualization_in_open_source_software.md) first, and then come back here.&&This module also assumes some basic familiarity with R, including&&* [installing and loading packages](https://r4ds.had.co.nz/data-visualisation.html#prerequisites-1)&* [reading in data](https://r4ds.had.co.nz/data-import.html)&* manipulating data frames, including [calculating new columns](https://r4ds.had.co.nz/transform.html#add-new-variables-with-mutate), and [pivoting from wide format to long](https://r4ds.had.co.nz/tidy-data.html#longer)&* some [statistical tests](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/statistical_tests/statistical_tests.md), especially linear regression&&If you are brand new to R (or want a refresher) consider starting with [Intro to R](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_introduction/r_basics_introduction.md) first.&&" 
df.loc["data_visualization_in_ggplot2", "Sets You Up For"] = "   r_practice  " 
df.loc["data_visualization_in_ggplot2", "Depends On Knowledge In"] = "   r_basics_introduction   data_visualization_in_open_source_software  " 
df.loc["data_visualization_in_open_source_software", "title"] = "Data Visualization in Open Source Software"
df.loc["data_visualization_in_open_source_software", "author"] = "Rose Hartman"
df.loc["data_visualization_in_open_source_software", "estimated_time_in_minutes"] = "20"
df.loc["data_visualization_in_open_source_software", "good_first_module"] = "false" 
df.loc["data_visualization_in_open_source_software", "sequence_name"] = "data_visualization"
df.loc["data_visualization_in_open_source_software", "data_task"] = "data_visualization"
df.loc["data_visualization_in_open_source_software", "comment"] = "Introduction to principles of data vizualization and typical data vizualization workflows using two common open source libraries: ggplot2 and seaborn." 
df.loc["data_visualization_in_open_source_software", "long_description"] = "This module introduces ggplot2 and seaborn, popular data visualization libraries in R and python, respectively. It lays the groundwork for using ggplot2 and seaborn by 1) highlighting common features of plots that can be manipulated in plot code, 2) discussing a typical data visualization workflow and best practices, and 3) discussing data preparation for plotting. This content will be most useful for people who have some experience creating data visualizations and/or reading plots presented in research articles or similar contexts. Some prior exposure to R and/or python is helpful but not required. This is appropriate for beginners." 
df.loc["data_visualization_in_open_source_software", "learning_objectives"] = "&After completion of this module, learners will be able to:&&* identify key elements in a plot that communicate information about the data&* describe the role ggplot2 and seaborn play in the R and python programming languages, respectively&* describe a typical data vizualization workflow&* list some best practices for creating accessible vizualizations&&" 
df.loc["data_visualization_in_open_source_software", "Prerequisties"] = "&This module assumes some familiarity with data and statistics, in particular&&* familiarity with some different kinds of plots, although deep understanding is not needed --- people who are used to seeing plots presented in research articles will be sufficiently prepared&* the distinction between [continuous and categorical variables](https://education.arcus.chop.edu/variable-types/)&&This module also assumes some basic familiarity with either R or python, but is appropriate for beginners.&&" 
df.loc["data_visualization_in_open_source_software", "Sets You Up For"] = "  data_visualization_in_seaborn   data_visualization_in_ggplot2 " 
df.loc["data_visualization_in_open_source_software", "Depends On Knowledge In"] = " " 
df.loc["data_visualization_in_seaborn", "title"] = "Data Visualization in seaborn"
df.loc["data_visualization_in_seaborn", "author"] = "Rose Hartman"
df.loc["data_visualization_in_seaborn", "estimated_time_in_minutes"] = "60"
df.loc["data_visualization_in_seaborn", "good_first_module"] = "false" 
df.loc["data_visualization_in_seaborn", "coding_required"] = "true"
df.loc["data_visualization_in_seaborn", "coding_language"] = "python"
df.loc["data_visualization_in_seaborn", "coding_level"] = "basic"
df.loc["data_visualization_in_seaborn", "sequence_name"] = "data_visualization"
df.loc["data_visualization_in_seaborn", "data_task"] = "data_visualization"
df.loc["data_visualization_in_seaborn", "comment"] = "This module includes code and explanations for several popular data visualizations using python's seaborn library. It also includes examples of how to modify seaborn plots to customize them for different uses.  " 
df.loc["data_visualization_in_seaborn", "long_description"] = "You can use the seaborn module in python to make many different kinds of data visualizations (also called plots or charts), including scatterplots, histograms, line plots, and trend lines. This module provides an example of each of these kinds of plots, including python code to make them using the seaborn module. It may be hard to follow if you are brand new to python, but it is appropriate for beginners with at least a small amount of python experience." 
df.loc["data_visualization_in_seaborn", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- use seaborn to create several common data visualizations&- customize some elements of a plot, and know where to look to learn how to customize others&&" 
df.loc["data_visualization_in_seaborn", "Prerequisties"] = "&This module assumes some familiarity with statistical concepts like distributions, outliers, and linear regression, but even if you don't understand those concepts well you should be able to learn and apply the data visualization content.&When statistical concepts are referenced in the lesson, links to learn more are generally provided.&&This module also assumes some basic familiarity with python, including&&* installing and importing python modules&* reading in data&* manipulating data frames, including calculating new columns&&If you are brand new to python (or want a refresher) consider starting with [Demystifying Python](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_python/demystifying_python.md) first.&&" 
df.loc["data_visualization_in_seaborn", "Sets You Up For"] = "   python_practice  " 
df.loc["data_visualization_in_seaborn", "Depends On Knowledge In"] = "   data_visualization_in_open_source_software   demystifying_python  " 
df.loc["database_normalization", "title"] = "Database Normalization"
df.loc["database_normalization", "author"] = "Joy Payton"
df.loc["database_normalization", "estimated_time_in_minutes"] = ""
df.loc["database_normalization", "comment"] = "Learn about the concept of normalization and why it's important for organizing complicated data in relational databases." 
df.loc["database_normalization", "long_description"] = "Usually, data in a relational database like SQL is organized into multiple interrelated tables with as little data repetition as possible. This concept can be useful to apply in other areas as well, such as organizing data in .csvs or in data frames in R or Python.  This module teaches underlying data considerations and explains how data can be efficiently organized by introducing the concepts of one-to-many data relationships and normalization." 
df.loc["database_normalization", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Explain the significance of +one to many+ data relationships and how these relationships affect data organization&- Describe how a normalized database is typically organized&- Explain how data can be linked between tables and define +primary keys+ and +foreign keys+&&" 
df.loc["demystifying_containers", "title"] = "Demystifying Containers"
df.loc["demystifying_containers", "author"] = "Meredith Lee"
df.loc["demystifying_containers", "estimated_time_in_minutes"] = "20"
df.loc["demystifying_containers", "good_first_module"] = "false" 
df.loc["demystifying_containers", "comment"] = "Containers can be a useful tool for reproducible workflows and collaboration. This module describes what containers are, why a researcher might want to use them, and what your options are for implementation. " 
df.loc["demystifying_containers", "long_description"] = "Writing code in multiple environments can be tricky. Collaboration can be hindered by different language versions, packages, and even operating systems. Developing code in containers can be a solution to this problem. In this module, we'll describe what containers are at a high level and discuss how they might be useful to researchers. " 
df.loc["demystifying_containers", "learning_objectives"] = "After completion of this module, learners will be able to:&&- understand when it might be useful to use containers for research&- describe the basic concept of containerization&- identify several containerization implementations&" 
df.loc["demystifying_containers", "Prerequisties"] = "The module assumes no prior familiarity with containers and requires no coding experience.  &" 
df.loc["demystifying_containers", "Sets You Up For"] = "  docker_101 " 
df.loc["demystifying_containers", "Depends On Knowledge In"] = " " 
df.loc["demystifying_geospatial_data", "title"] = "Demystifying Geospatial Data"
df.loc["demystifying_geospatial_data", "author"] = "Elizabeth Drellich"
df.loc["demystifying_geospatial_data", "estimated_time_in_minutes"] = "15"
df.loc["demystifying_geospatial_data", "good_first_module"] = "false" 
df.loc["demystifying_geospatial_data", "data_domain"] = "geospatial"
df.loc["demystifying_geospatial_data", "comment"] = "This module is a brief introduction to geospatial (location) data." 
df.loc["demystifying_geospatial_data", "long_description"] = "This module will survey some of the benefits of using geospatial data for research purposes. No previous exposure to geospatial data is expected. If you have any interest in maps or are wondering if using geospatial data might be helpful for your work, this lesson is designed to help you decide whether learning more about geospatial techniques is right for you and your project." 
df.loc["demystifying_geospatial_data", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define geospatial data&- Describe some of the benefits of using geospatial data&- Recognize some of the issues learners may encounter when using geospatial data&&" 
df.loc["demystifying_geospatial_data", "Prerequisties"] = "No prior knowledge or experience of geospatial data is required.&" 
df.loc["demystifying_geospatial_data", "Sets You Up For"] = "   geocode_lat_long  " 
df.loc["demystifying_geospatial_data", "Depends On Knowledge In"] = " " 
df.loc["demystifying_large_language_models", "title"] = "Demystifying Large Language Models"
df.loc["demystifying_large_language_models", "author"] = "Joy Payton"
df.loc["demystifying_large_language_models", "estimated_time_in_minutes"] = "60"
df.loc["demystifying_large_language_models", "good_first_module"] = "false " 
df.loc["demystifying_large_language_models", "comment"] = "Learn about large language models (LLM) like ChatGPT." 
df.loc["demystifying_large_language_models", "long_description"] = "There's lots of talk these days about large language models in academia, research, and medical circles.  What is a large language model, what can it actually do, and how might LLMs impact your career?  Learn more here!" 
df.loc["demystifying_large_language_models", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Define +large language model+ (LLM) &- Give a brief description of n-grams and word vectors&- Give a brief description of a neural network&- Give one example of a task that an LLM could do that could advance a biomedical project or career&- Give one example of a caveat or pitfall to be aware of when using an LLM&&" 
df.loc["demystifying_large_language_models", "Prerequisties"] = "None.  &" 
df.loc["demystifying_large_language_models", "Sets You Up For"] = " " 
df.loc["demystifying_large_language_models", "Depends On Knowledge In"] = " " 
df.loc["demystifying_machine_learning", "title"] = "Demystifying Machine Learning"
df.loc["demystifying_machine_learning", "author"] = "Rose Hartman"
df.loc["demystifying_machine_learning", "estimated_time_in_minutes"] = "60"
df.loc["demystifying_machine_learning", "good_first_module"] = "true" 
df.loc["demystifying_machine_learning", "comment"] = "An approachable and practical introduction to machine learning for biomedical researchers." 
df.loc["demystifying_machine_learning", "long_description"] = "If you're curious about machine learning and whether or not it could be useful to you in your work, this is for you. It provides a high-level overview of machine learning techniques with an emphasis on applications in biomedical research. This module covers the what and the why of machine learning only, not the how -- it doesn't include instructions or code for running models, just background to help you think about machine learning in research." 
df.loc["demystifying_machine_learning", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- list at least three potential applications of machine learning in biomedical science&- describe three different statistical problems models can address and how they differ (e.g. prediction, anomaly detection, clustering, dimension reduction)&- describe some potential pitfalls of machine learning and big data&&" 
df.loc["demystifying_machine_learning", "Prerequisties"] = "&This module assumes learners have been exposed to introductory statistics, like the distinction between [continuous and discrete variables](https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/random-variables-discrete/v/discrete-and-continuous-random-variables).&There are no coding exercises, and no programming experience is required.&&" 
df.loc["demystifying_machine_learning", "Sets You Up For"] = "   bias_variance_tradeoff  " 
df.loc["demystifying_machine_learning", "Depends On Knowledge In"] = " " 
df.loc["demystifying_python", "title"] = "Demystifying Python"
df.loc["demystifying_python", "author"] = "Meredith Lee"
df.loc["demystifying_python", "estimated_time_in_minutes"] = ""
df.loc["demystifying_python", "comment"] = "This module introduces the Python programming language, explores why Python is useful in research, and describes how to download Python and Jupyter." 
df.loc["demystifying_python", "long_description"] = "Python is a versatile programming language that is frequently used for data analysis, machine learning, web development, and more. If you are interested in using Python (or even just trying it out), and are looking for how to get set up, this module is a good place to start. This is appropriate for someone at the beginner level, including those with no prior knowledge of or experience with Python." 
df.loc["demystifying_python", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Describe what Python is and why they might want to use it for research&- Identify several ways to write Python code&- Understand the purpose and utility of a Jupyter notebook&- Download Python and Jupyter, and access a Python notebook in Google Colab&&" 
df.loc["demystifying_regular_expressions", "title"] = "Demystifying Regular Expressions"
df.loc["demystifying_regular_expressions", "author"] = "Joy Payton"
df.loc["demystifying_regular_expressions", "estimated_time_in_minutes"] = "30"
df.loc["demystifying_regular_expressions", "good_first_module"] = "false" 
df.loc["demystifying_regular_expressions", "coding_required"] = "true"
df.loc["demystifying_regular_expressions", "coding_level"] = "getting_started"
df.loc["demystifying_regular_expressions", "sequence_name"] = "regex"
df.loc["demystifying_regular_expressions", "comment"] = "Learn about pattern matching using regular expressions, or regex." 
df.loc["demystifying_regular_expressions", "long_description"] = "Regular expressions, or regex, are a way to specify patterns (such as the pattern that defines a valid email address, medical record number, or credit card number) within text.  Learn about how regular expressions can move your research forward in this non-coding module." 
df.loc["demystifying_regular_expressions", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Explain what a regular expression is &- Give an example of how regular expressions can be useful&- Use an online regular expressions checker that helps build and test regular expressions.&&" 
df.loc["demystifying_regular_expressions", "Prerequisties"] = "This module does not require any particular knowledge.  Anyone who has used a search function to find or find and replace text in a document will be able to engage with this content.&" 
df.loc["demystifying_regular_expressions", "Sets You Up For"] = "  regular_expressions_basics " 
df.loc["demystifying_regular_expressions", "Depends On Knowledge In"] = " " 
df.loc["demystifying_sql", "title"] = "Demystifying SQL"
df.loc["demystifying_sql", "author"] = "Peter Camacho"
df.loc["demystifying_sql", "estimated_time_in_minutes"] = ""
df.loc["demystifying_sql", "comment"] = "SQL is a relational database solution that has been around for decades.  Learn more about this technology at a high level, without having to write code." 
df.loc["demystifying_sql", "long_description"] = "Do you have colleagues who use SQL or refer to +databases+ or +the data warehouse+ and you're not sure what it all means?  This module will give you some very high level explanations to help you understand what SQL is and some basic concepts for working with it.  There is no code or hands-on application in this module, so it's appropriate for people who have zero experience and want an overview of SQL." 
df.loc["demystifying_sql", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define the acronym +SQL+&- Explain the basic organization of data in relational databases&- Explain what +relational+ means in the phrase +relational database+&- Give an example of what kinds of tasks SQL is ideal for&&" 
df.loc["directories_and_file_paths", "title"] = "Directories and File Paths"
df.loc["directories_and_file_paths", "author"] = "Meredith Lee"
df.loc["directories_and_file_paths", "estimated_time_in_minutes"] = ""
df.loc["directories_and_file_paths", "comment"] = "In this module, learners will explore what a directory is and how to describe the location of a file using its file path.   " 
df.loc["directories_and_file_paths", "long_description"] = "When doing data analysis in a programming language like R or Python, figuring out how to point the program to the file you need can be confusing. This module will help you learn about how files and folders are organized on your computer, how to describe the location of your file in a couple of different ways, and name files and folders in a descriptive and systematic way." 
df.loc["directories_and_file_paths", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Describe what a directory is&- Distinguish between a relative file path and an absolute file path&- Describe the location of a file using its file path&- Describe a few best practices and conventions of naming files and folders&&" 
df.loc["docker_101", "title"] = "Getting Started with Docker for Research"
df.loc["docker_101", "author"] = "Rose Hartman"
df.loc["docker_101", "estimated_time_in_minutes"] = "60"
df.loc["docker_101", "good_first_module"] = "false" 
df.loc["docker_101", "coding_required"] = "true"
df.loc["docker_101", "coding_language"] = "bash"
df.loc["docker_101", "coding_level"] = "intermediate"
df.loc["docker_101", "comment"] = "This tutorial combines a hands-on interactive Docker tutorial published by Docker Inc with an academic article outlining best practices for using Docker for research. " 
df.loc["docker_101", "long_description"] = "If you've been curious about how to use Docker for your research, this module is a great place to start. The Docker 101 tutorial is a popular, hands-on approach to learning Docker that will get you using containers right away, so you can learn by doing. To help you bridge the gap between basic Docker use and best practices for using Docker in research, we also link to an article outlining 10 rules to help you create great containers for research, and lists of ready-to-use Docker images for a variety of analysis workflows. This module includes running and editing commands in the terminal, so you'll need some basic familiarity with bash, but it is otherwise appropriate for beginners. No prior experience with Docker or containers is assumed. " 
df.loc["docker_101", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Use the command line to create and run a container from a Dockerfile&- Share containers &- Understand both technical requirements and best practices for writing Dockerfiles for use in research&" 
df.loc["docker_101", "Prerequisties"] = "This module assumes no prior experience with containers, and no particular coding other than some familiarity with the command line, such as being able to change directories and run bash commands that will be supplied for you to copy and paste. You will need to create and edit text files in a text editor like VSCode. &&You'll also need to create an account on [Docker Hub](https://hub.docker.com/) (it's free), if you don't have one already, and you'll need to be able to install the Docker Desktop software on your machine (also free). &" 
df.loc["docker_101", "Sets You Up For"] = " " 
df.loc["docker_101", "Depends On Knowledge In"] = "  demystifying_containers " 
df.loc["elements_of_maps", "title"] = "The Elements of Maps"
df.loc["elements_of_maps", "author"] = "Elizabeth Drellich"
df.loc["elements_of_maps", "estimated_time_in_minutes"] = ""
df.loc["elements_of_maps", "comment"] = "This is a general overview of ways that geospatial data can be communicated visually using maps." 
df.loc["elements_of_maps", "long_description"] = "Raw geospatial data can be particularly tricky for humans to read. However the shapes, colors, sizes, symbols, and language that make up a good map can effectively communicate a variety of detailed data even to readers looking at the map with only minimum specialized background knowledge. This module will demystify how raw data becomes a map and explain common components of maps. It is appropriate for anyone considering making maps from geospatial data." 
df.loc["elements_of_maps", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- recognize the elements of maps&- describe types of maps that focus on particular elements.&&" 
df.loc["genomics_quality_control", "title"] = "Genomics Tools and Methods: Quality Control"
df.loc["genomics_quality_control", "author"] = "Rose Hartman"
df.loc["genomics_quality_control", "estimated_time_in_minutes"] = "40"
df.loc["genomics_quality_control", "good_first_module"] = "false" 
df.loc["genomics_quality_control", "coding_required"] = "true"
df.loc["genomics_quality_control", "coding_language"] = "bash"
df.loc["genomics_quality_control", "coding_level"] = "intermediate"
df.loc["genomics_quality_control", "sequence_name"] = "genomics_tools_and_methods"
df.loc["genomics_quality_control", "data_domain"] = "omics"
df.loc["genomics_quality_control", "comment"] = "Get started with genomics! This module walks you through how to analyze FASTQ files to assess read quality, the first step in a common genomics workflow - identifying variants among sequencing samples taken from multiple individuals within a population (variant calling). " 
df.loc["genomics_quality_control", "long_description"] = "This module uses command line tools to complete the first steps of genomics analysis using cloud computing. We'll look at real sequencing data from an *E. coli* experiment and walk through how to assess the quality of sequenced reads using FastQC. You'll learn about FASTQ files and how to analyze them. This module assumes some familiarity with bash; if you've worked through some bash training already, this is a great opportunity to practice those skills while getting hands-on experience with genomics.  " 
df.loc["genomics_quality_control", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Explain how a FASTQ file encodes per-base quality scores.&- Interpret a FastQC plot summarizing per-base quality across all reads.&- Use `for` loops to automate operations on multiple files.&" 
df.loc["genomics_quality_control", "Prerequisties"] = "This lesson assumes a working understanding of the bash shell, including the following commands: `ls`, `cd`, `mkdir`, `grep`, `less`, `cat`, `ssh`, `scp`, and `for` loops.&If you aren’t familiar with the bash shell, please review our [Command Line 101](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_101/bash_command_line_101.md) module and/or the [Shell Genomics lesson by Data Carpentry](http://www.datacarpentry.org/shell-genomics/) before starting this lesson.&&This lesson also assumes some familiarity with biological concepts (including the structure of DNA, nucleotide abbreviations, and the concept of genomic variation within a population) and genomics (concepts like sequencing). &It does not assume any experience with genomics analysis. &" 
df.loc["genomics_quality_control", "Sets You Up For"] = " " 
df.loc["genomics_quality_control", "Depends On Knowledge In"] = "   bash_103_combining_commands   bash_command_line_101   bash_command_line_102   bash_conditionals_loops   data_storage_models   directories_and_file_paths   genomics_setup   omics_orientation  " 
df.loc["genomics_setup", "title"] = "Genomics Tools and Methods: Computing Setup"
df.loc["genomics_setup", "author"] = "Rose Hartman"
df.loc["genomics_setup", "estimated_time_in_minutes"] = "30"
df.loc["genomics_setup", "good_first_module"] = "false" 
df.loc["genomics_setup", "coding_required"] = "true"
df.loc["genomics_setup", "coding_language"] = "bash"
df.loc["genomics_setup", "coding_level"] = "intermediate"
df.loc["genomics_setup", "data_domain"] = "omics"
df.loc["genomics_setup", "comment"] = "This module walks you through setting up your own copy of a genomics analysis AMI (Amazon Machine Image) to run genomics analyses in the cloud. " 
df.loc["genomics_setup", "long_description"] = "One challenge to getting started with genomics is that it's often not feasible to run even basic analyses on a personal computer; to work with genomics data, you need to first set up a cloud computing environment that will support it. This module walks you through how to set up the AMI (Amazon Machine Image) published by Data Carpentry as part of their Genomics Workshop. " 
df.loc["genomics_setup", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Launch and terminate instances on AWS&- Use the Data Carpentry Community AMI to set up an AMI set up for genomics anlaysis&&" 
df.loc["genomics_setup", "Prerequisties"] = "This lesson assumes a working understanding of the bash shell.&If you aren’t familiar with the bash shell, please review our [Command Line 101 module](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/bash_command_line_101/bash_command_line_101.md) and/or the [Shell Genomics lesson by Data Carpentry](http://www.datacarpentry.org/shell-genomics/) before starting this lesson.&&" 
df.loc["genomics_setup", "Sets You Up For"] = " " 
df.loc["genomics_setup", "Depends On Knowledge In"] = " " 
df.loc["geocode_lat_long", "title"] = "Encoding Geospatial Data: Latitude and Longitude"
df.loc["geocode_lat_long", "author"] = "Elizabeth Drellich"
df.loc["geocode_lat_long", "estimated_time_in_minutes"] = ""
df.loc["geocode_lat_long", "comment"] = "This is an introduction to latitude and longitude and the importance of geocoding - encoding geospatial data in the coordinate system." 
df.loc["geocode_lat_long", "long_description"] = "If you use any geospatial data, such as patient or participant addresses, it is important that that location data be in a usable form. This means using the same coordinate system that Global Positioning Systems use: latitude and longitude. This module is appropriate, as either an introduction or review, for anyone considering using geospatial data in their analysis. " 
df.loc["geocode_lat_long", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Understand the importance of geocoding addresses&- Understand the latitude and longitude coordinate system&- Geocode single addresses.  &&" 
df.loc["git_creation_and_tracking", "title"] = "Creating a Git Repository"
df.loc["git_creation_and_tracking", "author"] = "Elizabeth Drellich"
df.loc["git_creation_and_tracking", "estimated_time_in_minutes"] = ""
df.loc["git_creation_and_tracking", "comment"] = "Create a new Git repository and get started with version control." 
df.loc["git_creation_and_tracking", "long_description"] = "If you have Git set up on your computer and are ready to start tracking your files, then this module is for you. This module will teach you how to create a Git repository, add files to it, update files in it, and keep track of those changes in a clear and organized manner." 
df.loc["git_creation_and_tracking", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Create a Git repository&- Add and make changes to files in the repository&- Write short helpful descriptions, called +commit messages+ to track the changes&- Use `.gitignore`&- Understand the `add` and `commit` workflow.&&&" 
df.loc["git_history_of_project", "title"] = "Exploring the History of your Git Repository"
df.loc["git_history_of_project", "author"] = "Elizabeth Drellich"
df.loc["git_history_of_project", "estimated_time_in_minutes"] = "30"
df.loc["git_history_of_project", "good_first_module"] = "false" 
df.loc["git_history_of_project", "coding_required"] = "true"
df.loc["git_history_of_project", "coding_language"] = "git, bash"
df.loc["git_history_of_project", "coding_level"] = "basic"
df.loc["git_history_of_project", "sequence_name"] = "git_basics"
df.loc["git_history_of_project", "comment"] = "This module will teach you how to look at past versions of your work on Git and compare your project with previous versions." 
df.loc["git_history_of_project", "long_description"] = "You know that version control is important. You know how to save your work to your Git repository. Now you are ready to look at and compare different versions of your work. In this module you will you will learn how to navigate through the commits you have made to Git. You will also learn how to compare current code with past code." 
df.loc["git_history_of_project", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Identify and use the `HEAD` of a repository.&- Identify and use Git commit numbers.&- Compare versions of tracked files.&&" 
df.loc["git_history_of_project", "Prerequisties"] = "To best learn from this module make sure that you:&&- have Git configured on your computer,&- can view and edit `.txt` files, and&- can make changes to a Git repository using `add` and `commit` from a command line interface (CLI).&&" 
df.loc["git_history_of_project", "Sets You Up For"] = " " 
df.loc["git_history_of_project", "Depends On Knowledge In"] = "  git_intro   git_setup_windows   git_setup_mac_and_linux   bash_command_line_101   git_creation_and_tracking " 
df.loc["git_intro", "title"] = "Intro to Version Control"
df.loc["git_intro", "author"] = "Rose Hartman"
df.loc["git_intro", "estimated_time_in_minutes"] = "15"
df.loc["git_intro", "good_first_module"] = "false" 
df.loc["git_intro", "coding_required"] = "false"
df.loc["git_intro", "sequence_name"] = "git_basics"
df.loc["git_intro", "comment"] = "An introduction to what version control systems do and why you might want to use one." 
df.loc["git_intro", "long_description"] = "Version control systems allow you to keep track of the history of changes to a text document (e.g. writing, code, and more). Version control is an increasingly important tool for scientists and scientific writers of all disciplines; it has the potential to make your work more transparent, more reproducible, and more efficient. This module is appropriate for beginners with no previous exposure to version control." 
df.loc["git_intro", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Understand the benefits of an automated version control system&- Understand the basics of how automated version control systems work&- Explain how git and GitHub differ&&" 
df.loc["git_intro", "Prerequisties"] = "&None. This lesson is appropriate for beginners with no experience using version control. Experience using word processing software like Microsoft Word, Google Docs, or LibreOffice may be helpful but is not required.&&" 
df.loc["git_setup_mac_and_linux", "title"] = "Setting Up Git on Mac and Linux"
df.loc["git_setup_mac_and_linux", "author"] = "Rose Hartman"
df.loc["git_setup_mac_and_linux", "estimated_time_in_minutes"] = ""
df.loc["git_setup_mac_and_linux", "comment"] = "This module provides recommendations and examples to help new users configure git on their computer for the first time on a Mac or Linux computer." 
df.loc["git_setup_mac_and_linux", "long_description"] = "If you're ready to start using the Git version control system, this lesson will walk you through how to get set up. This lesson should be a good fit for people who already have an idea of what version control is (although may not have any experience using it yet), and know how to open the command line interface (CLI) on their computer. No previous experience with Git is expected." 
df.loc["git_setup_mac_and_linux", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Configure `git` the first time it is used on a computer&- Understand the meaning of the `--global` configuration flag&&" 
df.loc["git_setup_windows", "title"] = "Setting Up Git on Windows"
df.loc["git_setup_windows", "author"] = "Elizabeth Drellich"
df.loc["git_setup_windows", "estimated_time_in_minutes"] = ""
df.loc["git_setup_windows", "comment"] = "This module provides recommendations and examples to help new users configure Git on their Windows computer for the first time." 
df.loc["git_setup_windows", "long_description"] = "If you're ready to start using the Git version control system, this lesson will walk you through how to get set up. This lesson should be a good fit for people who already have an idea of what version control is but may not have any experience using it yet. No previous experience with Git is expected. This lesson is specific to Windows machines, If you are using Mac or Linux, please follow along with the [set-up guide for those computers](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/git_setup_mac_and_linux/git_setup_mac_and_linux.md)." 
df.loc["git_setup_windows", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Configure `git` the first time it is used on a computer&- Understand the meaning of the `--global` configuration flag&&" 
df.loc["how_to_troubleshoot", "title"] = "How to Troubleshoot"
df.loc["how_to_troubleshoot", "author"] = "Joy Payton"
df.loc["how_to_troubleshoot", "estimated_time_in_minutes"] = ""
df.loc["how_to_troubleshoot", "comment"] = "Learning to use technical methods like coding and version control in your research inevitably means running into problems.  Learn practical methods for troubleshooting and moving past error codes and other difficulties." 
df.loc["how_to_troubleshoot", "long_description"] = "When technical methods, such as writing code, using version control, and creating data visualizations are used, there will moments when a cryptic error message appears or the code simply doesn't do what it was intended to do.  This module will help people at various levels of technical expertise learn how to troubleshoot in tech more effectively." 
df.loc["how_to_troubleshoot", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Describe technical problems more effectively&- Explain why a +reproducible example+ is critical to asking for help&- Find potentially helpful answers in Stack Overflow&&&" 
df.loc["intro_to_nhst", "title"] = "Introduction to Null Hypothesis Significance Testing"
df.loc["intro_to_nhst", "author"] = "Rose Hartman"
df.loc["intro_to_nhst", "estimated_time_in_minutes"] = "40"
df.loc["intro_to_nhst", "good_first_module"] = "false" 
df.loc["intro_to_nhst", "data_task"] = "data_analysis"
df.loc["intro_to_nhst", "comment"] = "This is an introduction to NHST for biomedical researchers. " 
df.loc["intro_to_nhst", "long_description"] = "Null Hypothesis Significance Testing (NHST) is by far the most commonly used method of statistical inference in research --- regression, ANOVAs, and t-tests are all tests from the NHST framework. This module introduces the important concepts that underlie NHST and prepares you to learn how to use NHST responsibly in your research. It does not assume any prior knowledge of statistics. " 
df.loc["intro_to_nhst", "learning_objectives"] = "After completion of this module, learners will be able to:&&- identify the null hypothesis given a research question&- define a p-value&- define Type 1 error, Type 2 error, and statistical power&- describe common pitfalls of NHST in research and how to avoid them&&" 
df.loc["intro_to_nhst", "Prerequisties"] = "None.&" 
df.loc["intro_to_nhst", "Sets You Up For"] = "   statistical_tests  " 
df.loc["intro_to_nhst", "Depends On Knowledge In"] = " " 
df.loc["learning_to_learn", "title"] = "Learning to Learn Data Science"
df.loc["learning_to_learn", "author"] = "Rose Franzen"
df.loc["learning_to_learn", "estimated_time_in_minutes"] = ""
df.loc["learning_to_learn", "comment"] = "Discover how learning data science is different than learning other subjects." 
df.loc["learning_to_learn", "long_description"] = "The process of learning data science can be different from that of learning other subjects. This module goes over some of those differences and provides advice for navigating this potentially unfamiliar territory." 
df.loc["learning_to_learn", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- recognize ways in which learning data science and coding may be different than other educational experiences&- identify ways to extend their learning beyond module content&- recognize how to understand when to ask for help&&" 
df.loc["omics_orientation", "title"] = "Omics Orientation"
df.loc["omics_orientation", "author"] = "Meredith Lee"
df.loc["omics_orientation", "estimated_time_in_minutes"] = ""
df.loc["omics_orientation", "comment"] = "This module provides a brief introduction to omics and its associated fields." 
df.loc["omics_orientation", "long_description"] = "Omics is a wide-reaching field, with many different subfields. This module aims to disambiguate several omics-related terms and topics, discuss some of the most popular omics research fields, and examine the challenges of and caveats for omics research." 
df.loc["omics_orientation", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define what omics is and explain why a researcher might choose an omics approach&- Identify several popular omics domains&- Describe some challenges and caveats of omics research&&&" 
df.loc["pandas_transform", "title"] = "Transform Data with pandas"
df.loc["pandas_transform", "author"] = "Elizabeth Drellich"
df.loc["pandas_transform", "estimated_time_in_minutes"] = "60"
df.loc["pandas_transform", "good_first_module"] = "false" 
df.loc["pandas_transform", "coding_required"] = "true"
df.loc["pandas_transform", "coding_language"] = "python"
df.loc["pandas_transform", "coding_level"] = "intermediate"
df.loc["pandas_transform", "data_task"] = "data_wrangling"
df.loc["pandas_transform", "comment"] = "This is an introduction to transforming data using a Python library named pandas." 
df.loc["pandas_transform", "long_description"] = "This module is for learners who have some familiarity with Python, and want to learn how the pandas library can handle large tabular data sets. No previous experience with pandas is required, and only an introductory level understanding of Python is assumed." 
df.loc["pandas_transform", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Import `pandas` and use functions from the `pandas` package.&- Load data into a `pandas` DataFrame.&- Use the `.loc` method to explore the contents of a DataFrame&- Filter a DataFrame using conditional statements.&- Transform data in a DataFrame.&&" 
df.loc["pandas_transform", "Prerequisties"] = "Before starting this module it is useful for you to:&&- have some familiarity with tabular data: data stored in an array of rows and columns.&&- have an introductory level exposure to coding in Python, which could be acquired in the Python Basics sequence of modules ([Functions, Methods, and Variables](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#1); [Lists and Dictionaries](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md#1); and [Loops and Conditional Statements](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_loops_conditionals/python_basics_loops_conditionals.md#1)).&" 
df.loc["pandas_transform", "Sets You Up For"] = "   python_practice  " 
df.loc["pandas_transform", "Depends On Knowledge In"] = "   python_basics_variables_functions_methods   python_basics_lists_dictionaries   python_basics_loops_conditionals  " 
df.loc["python_basics_exercise", "title"] = "Python Basics: Exercise"
df.loc["python_basics_exercise", "author"] = "Meredith Lee"
df.loc["python_basics_exercise", "estimated_time_in_minutes"] = "30"
df.loc["python_basics_exercise", "good_first_module"] = "false" 
df.loc["python_basics_exercise", "coding_required"] = "true"
df.loc["python_basics_exercise", "coding_language"] = "python"
df.loc["python_basics_exercise", "coding_level"] = "basic"
df.loc["python_basics_exercise", "sequence_name"] = "python_basics"
df.loc["python_basics_exercise", "comment"] = "Practice the skills acquired in the Python Basics sequence by working through an exercise. " 
df.loc["python_basics_exercise", "long_description"] = "Now that you've learned a bit about the basics of Python programming, it's time to try to put these concepts together! This module presents an exercise that can be solved using the skills you've learned in the Python Basics sequence (using [functions](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#5), [methods](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#6), [variables](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#9), [lists](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md#4), [dictionaries](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md#6), [loops](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_loops_conditionals/python_basics_loops_conditionals.md#4), and [conditional statements](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_loops_conditionals/python_basics_loops_conditionals.md#8))." 
df.loc["python_basics_exercise", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Run their own Python code, either on their own computer or in the cloud.&- Loop through a dictionary and conditionally perform an iterative task based on the values in the dictionary. &&" 
df.loc["python_basics_exercise", "Prerequisties"] = "Learners should be familiar with using functions, methods, variables, lists, dictionaries, loops, and conditional statements in Python. These skills are presented in the Python Basics sequence of modules ([Functions, Methods, and Variables](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#1), [Lists and Dictionaries](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md#1), and [Loops and Conditional Statements](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_loops_conditionals/python_basics_loops_conditionals.md#1)).&" 
df.loc["python_basics_exercise", "Sets You Up For"] = " " 
df.loc["python_basics_exercise", "Depends On Knowledge In"] = "   demystifying_python   python_basics_variables_functions_methods   python_basics_lists_dictionaries   python_basics_loops_conditionals  " 
df.loc["python_basics_lists_dictionaries", "title"] = "Python Basics: Lists and Dictionaries"
df.loc["python_basics_lists_dictionaries", "author"] = "Meredith Lee"
df.loc["python_basics_lists_dictionaries", "estimated_time_in_minutes"] = "15"
df.loc["python_basics_lists_dictionaries", "good_first_module"] = "false" 
df.loc["python_basics_lists_dictionaries", "coding_required"] = "true"
df.loc["python_basics_lists_dictionaries", "coding_language"] = "python"
df.loc["python_basics_lists_dictionaries", "coding_level"] = "basic"
df.loc["python_basics_lists_dictionaries", "sequence_name"] = "python_basics"
df.loc["python_basics_lists_dictionaries", "comment"] = "Learn about collection objects, specifically lists and dictionaries, in Python." 
df.loc["python_basics_lists_dictionaries", "long_description"] = "Before using Python for data analysis, there are some basics to learn that will set the foundation for more advanced Python coding. This module will teach you about lists and dictionaries, two types of collection objects in Python. " 
df.loc["python_basics_lists_dictionaries", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Create and edit lists&- Create and edit dictionaries&&" 
df.loc["python_basics_lists_dictionaries", "Prerequisties"] = "Learners should be able to recognize functions, methods, and variables in Python.&" 
df.loc["python_basics_lists_dictionaries", "Sets You Up For"] = "   python_basics_loops_conditionals   python_basics_exercise   pandas_transform  " 
df.loc["python_basics_lists_dictionaries", "Depends On Knowledge In"] = "   demystifying_python   python_basics_variables_functions_methods  " 
df.loc["python_basics_loops_conditionals", "title"] = "Python Basics: Loops and Conditionals"
df.loc["python_basics_loops_conditionals", "author"] = "Meredith Lee"
df.loc["python_basics_loops_conditionals", "estimated_time_in_minutes"] = "20"
df.loc["python_basics_loops_conditionals", "good_first_module"] = "false" 
df.loc["python_basics_loops_conditionals", "coding_required"] = "true"
df.loc["python_basics_loops_conditionals", "coding_language"] = "python"
df.loc["python_basics_loops_conditionals", "coding_level"] = "basic"
df.loc["python_basics_loops_conditionals", "sequence_name"] = "python_basics"
df.loc["python_basics_loops_conditionals", "comment"] = "Learn how to use loops and conditional statements in Python. " 
df.loc["python_basics_loops_conditionals", "long_description"] = "Before using Python for data analysis, there are some basics to learn that will set the foundation for more advanced Python coding. This module will teach you about how to loop through sequences and use conditional statements. " 
df.loc["python_basics_loops_conditionals", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Iterate through lists using loops&- Utilize conditional statements&&" 
df.loc["python_basics_loops_conditionals", "Prerequisties"] = "Learners should be familiar with using [functions and methods](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md#1) and [collections](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md#1) at a beginner level. &" 
df.loc["python_basics_loops_conditionals", "Sets You Up For"] = "   python_basics_exercise   pandas_transform  " 
df.loc["python_basics_loops_conditionals", "Depends On Knowledge In"] = "   demystifying_python   python_basics_variables_functions_methods   python_basics_lists_dictionaries  " 
df.loc["python_basics_variables_functions_methods", "title"] = "Python Basics: Functions, Methods, and Variables"
df.loc["python_basics_variables_functions_methods", "author"] = "Meredith Lee"
df.loc["python_basics_variables_functions_methods", "estimated_time_in_minutes"] = "20"
df.loc["python_basics_variables_functions_methods", "good_first_module"] = "false" 
df.loc["python_basics_variables_functions_methods", "coding_required"] = "true"
df.loc["python_basics_variables_functions_methods", "coding_language"] = "python"
df.loc["python_basics_variables_functions_methods", "coding_level"] = "basic"
df.loc["python_basics_variables_functions_methods", "sequence_name"] = "python_basics"
df.loc["python_basics_variables_functions_methods", "comment"] = "Learn the foundations of writing Python code, including the use of functions, methods, and variables." 
df.loc["python_basics_variables_functions_methods", "long_description"] = "Before using Python for data analysis, there are some basics to learn that will set the foundation for more advanced Python coding. This module will teach you about how to define variables and how to use functions and methods. " 
df.loc["python_basics_variables_functions_methods", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Assign values to variables&- Identify and use functions &- Identify and use methods&&" 
df.loc["python_basics_variables_functions_methods", "Prerequisties"] = "Learners should be familiar with [Python as a programming language](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_python/demystifying_python.md), but experience with writing Python code is not required.&" 
df.loc["python_basics_variables_functions_methods", "Sets You Up For"] = "   python_basics_dictionaries   python_basics_loops_conditionals   python_basics_exercise  " 
df.loc["python_basics_variables_functions_methods", "Depends On Knowledge In"] = "   demystifying_python  " 
df.loc["python_practice", "title"] = "Python Practice"
df.loc["python_practice", "author"] = "Meredith Lee"
df.loc["python_practice", "estimated_time_in_minutes"] = "60"
df.loc["python_practice", "good_first_module"] = "false" 
df.loc["python_practice", "coding_required"] = "true"
df.loc["python_practice", "coding_language"] = "python"
df.loc["python_practice", "coding_level"] = "intermediate"
df.loc["python_practice", "comment"] = "Use the basics of Python coding, data transformation, and data visualization to work with real data. " 
df.loc["python_practice", "long_description"] = "When learning Python for data science, the ultimate goal is to be able to put all of the pieces together to analyze a dataset. This module aims to provide a data science task in order to help learners practice Python skills in a real-world context. " 
df.loc["python_practice", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Import a dataset from an online database&- Recode data and change variable types in a dataframe&- Use exploratory data visualization to identify trends in data and generate hypotheses&&" 
df.loc["python_practice", "Prerequisties"] = "Learners should be familiar with the basics of Python coding, including [functions, methods, and variables](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_variables_functions_methods/python_basics_variables_functions_methods.md), [lists and dictionaries](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_lists_dictionaries/python_basics_lists_dictionaries.md), [loops and conditionals](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/python_basics_loops_conditionals/python_basics_loops_conditionals.md), [data transformation with pandas](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/pandas_transform/pandas_transform.md) and [data visualization with matplotlib and seaborn](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/data_visualization_in_seaborn/data_visualization_in_seaborn.md). Learners should also have access to Python, either on their own computer or in the cloud. &" 
df.loc["python_practice", "Sets You Up For"] = " " 
df.loc["python_practice", "Depends On Knowledge In"] = "   python_basics_variables_functions_methods   python_basics_lists_dictionaries   python_basics_loops_conditionals   pandas_transform   data_visualization_in_seaborn  " 
df.loc["r_basics_introduction", "title"] = "R Basics: Introduction"
df.loc["r_basics_introduction", "author"] = "Joy Payton"
df.loc["r_basics_introduction", "estimated_time_in_minutes"] = "60"
df.loc["r_basics_introduction", "good_first_module"] = "true" 
df.loc["r_basics_introduction", "coding_required"] = "true"
df.loc["r_basics_introduction", "coding_language"] = "r"
df.loc["r_basics_introduction", "coding_level"] = "basic"
df.loc["r_basics_introduction", "sequence_name"] = "r_basics"
df.loc["r_basics_introduction", "data_task"] = "data_analysis"
df.loc["r_basics_introduction", "comment"] = "Introduction to R and hands-on first steps for brand new beginners." 
df.loc["r_basics_introduction", "long_description"] = "Are you brand new to R, and ready to get started?  This module teaches concepts and vocabulary related to R, RStudio, and R Markdown.  It also includes some introductory-level hands-on work in RStudio.  This is a good course if you know that you want to use R but haven't ever used it, or you've barely used it and need a refresher on the basics." 
df.loc["r_basics_introduction", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define and differentiate +R+, +RStudio+, and +R Markdown+&- Install and load packages in R&- Create a simple R Markdown file and its associated output document&- Import a .csv file as a data frame&&" 
df.loc["r_basics_introduction", "Prerequisties"] = "&No prior experience of using R, RStudio, or R Markdown is required for this course.   &&This course is designed for brand new beginners with zero or minimal experience working with R.&&" 
df.loc["r_basics_introduction", "Sets You Up For"] = "   r_basics_transform_data   r_basics_visualize_data   r_missing_values   r_practice   r_reshape_lonog_wide   r_summary_stats   data_visualization_in_ggplot2  " 
df.loc["r_basics_introduction", "Depends On Knowledge In"] = " " 
df.loc["r_basics_transform_data", "title"] = "R Basics: Transforming Data With dplyr"
df.loc["r_basics_transform_data", "author"] = "Joy Payton"
df.loc["r_basics_transform_data", "estimated_time_in_minutes"] = "60"
df.loc["r_basics_transform_data", "good_first_module"] = "false" 
df.loc["r_basics_transform_data", "coding_required"] = "true"
df.loc["r_basics_transform_data", "coding_language"] = "r"
df.loc["r_basics_transform_data", "coding_level"] = "basic"
df.loc["r_basics_transform_data", "sequence_name"] = "r_basics"
df.loc["r_basics_transform_data", "data_task"] = "data_wrangling"
df.loc["r_basics_transform_data", "comment"] = "Learn how to transform (or wrangle) data using R's `dplyr` package." 
df.loc["r_basics_transform_data", "long_description"] = "Do you want to learn how to work with tabular (table-shaped, with rows and columns) data in R?  In this module you'll learn in particular how to select just the rows and columns you want to work with, how to create new columns, and how to create multi-step transformations to get your data ready for visualization or statistical analysis.  This module teaches the use of the `dplyr` package, which is part of the `tidyverse` suite of packages." 
df.loc["r_basics_transform_data", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Write R code that uses the `dplyr` package to select only desired columns from a data frame&- Write R code that uses the `dplyr` package to filter only rows that meet a certain condition from a data frame&- Write R code that uses the `dplyr` package to create a new column in a data frame&&" 
df.loc["r_basics_transform_data", "Prerequisties"] = "&Minimal experience of using the RStudio IDE and writing R code (specifically, within an R Markdown document) is necessary to understand and use this material.  If you can understand and do the following, you'll be able to complete this course:&&* Run a command that's provided to you in the console&* Use the Environment tab to find a data frame and learn more about it&* Insert a new code chunk in an R Markdown document&&" 
df.loc["r_basics_transform_data", "Sets You Up For"] = "  r_missing_values   r_practice   r_reshape_long_wide   r_summary_stats   data_visualization_in_ggplot2  " 
df.loc["r_basics_transform_data", "Depends On Knowledge In"] = "  r_basics_introduction   r_basics_visualize_data  " 
df.loc["r_basics_visualize_data", "title"] = "R Basics: Visualizing Data With ggplot2"
df.loc["r_basics_visualize_data", "author"] = "Joy Payton"
df.loc["r_basics_visualize_data", "estimated_time_in_minutes"] = "60"
df.loc["r_basics_visualize_data", "good_first_module"] = "false" 
df.loc["r_basics_visualize_data", "coding_required"] = "true"
df.loc["r_basics_visualize_data", "coding_language"] = "r"
df.loc["r_basics_visualize_data", "coding_level"] = "basic"
df.loc["r_basics_visualize_data", "sequence_name"] = "r_basics"
df.loc["r_basics_visualize_data", "data_task"] = "data_visualization"
df.loc["r_basics_visualize_data", "comment"] = "Learn how to visualize data using R's `ggplot2` package." 
df.loc["r_basics_visualize_data", "long_description"] = "Do you want to learn how to make some basic data visualizations (graphs) in R?  In this module you'll learn about the +grammar of graphics+ and the base code that you need to get started.  We'll use the basic ingredients of a tidy data frame, a geometric type, and some aesthetic mappings (we'll explain what all of those are).  This module teaches the use of the `ggplot2` package, which is part of the `tidyverse` suite of packages." 
df.loc["r_basics_visualize_data", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Write R code that creates basic data visualizations&- Identify geometric plot types available in `ggplot2`&- Map columns of data to visual elements like color or position&&" 
df.loc["r_basics_visualize_data", "Prerequisties"] = "&Minimal experience of using the RStudio IDE and writing R code (specifically, within an R Markdown document) is necessary to understand and use this material.  If you can understand and do the following, you'll be able to complete this course:&&* Run a command that's provided to you in the console&* Use the Environment tab to find a data frame and learn more about it&* Insert a new code chunk in an R Markdown document&&One potential way to get these basic skills is to take our [R Basics: Introduction](https://liascript.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_introduction/r_basics_introduction.md) course.&&This course is designed for R beginners with minimal experience and it is not an advanced course in `ggplot2`.  If you have experience with `ggplot2` already, you may find our [+Data Visualization in ggplot2+](https://liascript.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/data_visualization_in_ggplot2/data_visualization_in_ggplot2.md), which is more advanced, a better fit for your needs.&&" 
df.loc["r_basics_visualize_data", "Sets You Up For"] = "   r_practice  " 
df.loc["r_basics_visualize_data", "Depends On Knowledge In"] = "  r_basics_introduction  " 
df.loc["r_missing_values", "title"] = "Missing Values in R"
df.loc["r_missing_values", "author"] = "Rose Hartman"
df.loc["r_missing_values", "estimated_time_in_minutes"] = "45"
df.loc["r_missing_values", "good_first_module"] = "false" 
df.loc["r_missing_values", "coding_required"] = "true"
df.loc["r_missing_values", "coding_language"] = "r"
df.loc["r_missing_values", "coding_level"] = "basic"
df.loc["r_missing_values", "data_task"] = "data_wrangling"
df.loc["r_missing_values", "comment"] = "A practical demonstration of how missing values show up in R and how to deal with them. Note that this module does **not** cover statistical approaches for handling missing data, but instead focuses on the code you need to find, work with, and assign missing values in R." 
df.loc["r_missing_values", "long_description"] = "This is a beginner's guide to handling missing values in R. It covers what `NA` values are and how to check for them, how to mark values as missing, how to work around missing values in your analysis, and how to filter your data to remove missingness. This module is appropriate for learners who feel comfortable with R basics and are ready to get into the realities of data analysis, like dealing with missing values." 
df.loc["r_missing_values", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- check the number and location of missing values in a dataframe&- mark values as missing&- use common arguments like `na.rm` and `na.action` to control how functions handle missingness&- remove cases with missing values from a dataframe&&" 
df.loc["r_missing_values", "Prerequisties"] = "&This module assumes familiarity with R basics, including using dplyr tools to do basic transformation including choosing only certain columns or rows of a data frame and changing or adding columns.  &If you need to learn these basics, we suggest our [R Basics: Introduction](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_introduction/r_basics_introduction.md) module and our [R Basics: Transform Data](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_transform_data/r_basics_transform_data.md) module.&&This module also uses several R functions as examples to demonstrate how missing values work across a variety of settings, but it's fine if you don't have any experience with the example functions used.&When example functions for plotting or statistical tests appear in the module, there will be links provided for you to learn more about those functions if you're curious about them, but it's also fine to ignore that and just focus on how the missing values are handled.   &&" 
df.loc["r_missing_values", "Sets You Up For"] = "   r_practice  " 
df.loc["r_missing_values", "Depends On Knowledge In"] = "  r_basics_introduction  r_basics_transform_data  " 
df.loc["r_practice", "title"] = "R Practice"
df.loc["r_practice", "author"] = "Meredith Lee"
df.loc["r_practice", "estimated_time_in_minutes"] = ""
df.loc["r_practice", "comment"] = "Use the basics of R coding, data transformation, and data visualization to work with real data." 
df.loc["r_practice", "long_description"] = "When learning R for data science, the ultimate goal is to be able to put all of the pieces together to analyze a dataset. This module aims to provide a data science task in order to help learners practice R skills in a real-world context." 
df.loc["r_practice", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Import a dataset from an online database&- Recode data and change variable types in a dataframe&- Use exploratory data visualization to identify trends in data and generate hypotheses&&" 
df.loc["r_reshape_long_wide", "title"] = "Reshaping Data in R: Long and Wide Data"
df.loc["r_reshape_long_wide", "author"] = "Joy Payton"
df.loc["r_reshape_long_wide", "estimated_time_in_minutes"] = "60"
df.loc["r_reshape_long_wide", "good_first_module"] = "false" 
df.loc["r_reshape_long_wide", "coding_required"] = "true"
df.loc["r_reshape_long_wide", "coding_language"] = "r"
df.loc["r_reshape_long_wide", "coding_level"] = "intermediate"
df.loc["r_reshape_long_wide", "data_task"] = "data_wrangling"
df.loc["r_reshape_long_wide", "comment"] = "A module that teaches how to reshape tabular data in R, concentrating on some typical shapes known as +long+ and +wide+ data." 
df.loc["r_reshape_long_wide", "long_description"] = "Reshaping data is one of the essential skills in getting your data in a tidy format, ready to visualize, analyze, and model.  This module is appropriate for learners who feel comfortable with R basics and are ready to take on the challenges of real life data, which is often messy and requires considerable effort to tidy." 
df.loc["r_reshape_long_wide", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define and differentiate +long data+ and +wide data+&- Use tidyr and dplyr tools to reshape data effectively&&" 
df.loc["r_reshape_long_wide", "Prerequisties"] = "&This module assumes familiarity with R basics, including ingesting .csv data and using dplyr tools to do basic transformation including choosing only certain columns or rows of a data frame.  If you need to learn these basics, we suggest our [R Basics: Introduction](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_introduction/r_basics_introduction.md) module and our [R Basics: Transform Data](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_transform_data/r_basics_transform_data.md) module.&&" 
df.loc["r_reshape_long_wide", "Sets You Up For"] = "   r_practice  " 
df.loc["r_reshape_long_wide", "Depends On Knowledge In"] = "   r_basics_introduction   r_basics_transform_data  " 
df.loc["r_summary_stats", "title"] = "Summary Statistics in R"
df.loc["r_summary_stats", "author"] = "Rose Hartman"
df.loc["r_summary_stats", "estimated_time_in_minutes"] = "30"
df.loc["r_summary_stats", "good_first_module"] = "false" 
df.loc["r_summary_stats", "coding_required"] = "true"
df.loc["r_summary_stats", "coding_language"] = "r"
df.loc["r_summary_stats", "coding_level"] = "intermediate"
df.loc["r_summary_stats", "data_task"] = "data_analysis"
df.loc["r_summary_stats", "comment"] = "Learn to calculate summary statistics in R, and how to present them in a table for publication." 
df.loc["r_summary_stats", "long_description"] = "Get started with data analysis in R! This module covers how to get basic descriptive statistics for both continuous and categorical variables in R, and how to present your summary statistics in a beautiful publication-ready table (i.e. Table 1 in most research papers). You'll learn how to use several functions from base R including `summary()` to get a quick overview of your data, and also how to use the popular `gtsummary` package to create tables. " 
df.loc["r_summary_stats", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- calculate common summary statistics in R, for both continuous and categorical variables&- generate publication-ready tables of descriptive statistics using the gtsummary package&&" 
df.loc["r_summary_stats", "Prerequisties"] = "Minimal experience of using the RStudio IDE and writing R code (specifically, within an R Markdown document) is necessary to understand and use this material.  In particular, you should be able to do the following:&&* Run a command that's provided to you in the console&* Use the Environment tab to find a data frame and learn more about it&* Insert a new code chunk in an R Markdown document&&This module also assumes some basic familiarity with R, including&&* [installing and loading packages](https://r4ds.had.co.nz/data-visualisation.html#prerequisites-1)&* manipulating data frames, including [selecting columns and calculating new columns](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_transform_data/r_basics_transform_data.md)&* the difference between [numeric (continuous) and factor (categorical) variables](https://swcarpentry.github.io/r-novice-inflammation/13-supp-data-structures) in a dataframe&&If you are brand new to R (or want a refresher) consider starting with [Intro to R](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/r_basics_introduction/r_basics_introduction.md) first.&" 
df.loc["r_summary_stats", "Sets You Up For"] = " " 
df.loc["r_summary_stats", "Depends On Knowledge In"] = "r_basics_introduction r_basics_transform_data " 
df.loc["regular_expressions_basics", "title"] = "Regular Expressions Basics"
df.loc["regular_expressions_basics", "author"] = "Joy Payton"
df.loc["regular_expressions_basics", "estimated_time_in_minutes"] = "60"
df.loc["regular_expressions_basics", "good_first_module"] = "false" 
df.loc["regular_expressions_basics", "coding_required"] = "true"
df.loc["regular_expressions_basics", "coding_level"] = "basic"
df.loc["regular_expressions_basics", "sequence_name"] = "regex"
df.loc["regular_expressions_basics", "comment"] = "Begin to use regular expressions, or regex, for simple pattern matching." 
df.loc["regular_expressions_basics", "long_description"] = "Regular expressions, or regex, are a way to specify patterns (such as the pattern that defines a valid email address, medical record number, or credit card number).  Learn to compose basic regular expressions in order to find and use important data." 
df.loc["regular_expressions_basics", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Define a simple alphanumeric pattern in regex notation&- List common ranges and character groups in regex&- Quantify characters appearing optionally, once, or multiple times in regex&&" 
df.loc["regular_expressions_basics", "Prerequisties"] = "Learners should have some knowledge about patterns in biomedical data and understand the utility of regular expressions (regex).  For an introduction to these concepts, consider the [Demystifying Regular Expressions](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_regular_expressions/demystifying_regular_expressions.md#1) module. Having previously encountered an online regular expression checker may also be useful.&" 
df.loc["regular_expressions_basics", "Sets You Up For"] = " " 
df.loc["regular_expressions_basics", "Depends On Knowledge In"] = "  demystifying_regular_expressions " 
df.loc["regular_expressions_boundaries_anchors", "title"] = "Regular Expressions: Flags, Anchors, and Boundaries"
df.loc["regular_expressions_boundaries_anchors", "author"] = "Joy Payton"
df.loc["regular_expressions_boundaries_anchors", "estimated_time_in_minutes"] = "45"
df.loc["regular_expressions_boundaries_anchors", "good_first_module"] = "false" 
df.loc["regular_expressions_boundaries_anchors", "coding_required"] = "true"
df.loc["regular_expressions_boundaries_anchors", "coding_level"] = "intermediate"
df.loc["regular_expressions_boundaries_anchors", "sequence_name"] = "regex"
df.loc["regular_expressions_boundaries_anchors", "comment"] = "Use flags, anchors, and boundaries in regular expressions, or regex, for complex pattern matching." 
df.loc["regular_expressions_boundaries_anchors", "long_description"] = "Learn to compose intermediate regular expressions using flags, anchors, boundaries, and more, in order to find text that matches patterns you describe." 
df.loc["regular_expressions_boundaries_anchors", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Explain what a regular expression flag does&- Use anchors and boundaries in regular expressions&- Use boundaries in regular expressions&&" 
df.loc["regular_expressions_boundaries_anchors", "Prerequisties"] = "Learners should have some experience composing and using moderately complex regular expressions (regex).  For an introduction to regular expression concepts, consider the [Demystifying Regular Expressions](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_regular_expressions/demystifying_regular_expressions.md#1) module.  To learn how to compose and use simple regular expressions, consider the [Regular Expressions Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_basics/regular_expressions_basics.md#1) module.  We refer to groups in this module, which are covered in the [Regular Expressions: Groups](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_groups/regular_expressions_groups.md#1) module. &" 
df.loc["regular_expressions_boundaries_anchors", "Sets You Up For"] = "" 
df.loc["regular_expressions_boundaries_anchors", "Depends On Knowledge In"] = "  demystifying_regular_expressions   regular_expressions_basics   regular_expressions_groups " 
df.loc["regular_expressions_groups", "title"] = "Regular Expressions: Groups"
df.loc["regular_expressions_groups", "author"] = "Joy Payton"
df.loc["regular_expressions_groups", "estimated_time_in_minutes"] = "30"
df.loc["regular_expressions_groups", "good_first_module"] = "false" 
df.loc["regular_expressions_groups", "coding_required"] = "true"
df.loc["regular_expressions_groups", "coding_level"] = "intermediate"
df.loc["regular_expressions_groups", "sequence_name"] = "regex"
df.loc["regular_expressions_groups", "comment"] = "Use regular expressions, or regex, for complex pattern matching involving capturing and non-capturing groups." 
df.loc["regular_expressions_groups", "long_description"] = "Learn to compose advanced regular expressions using capturing and non-capturing groups in order to find and extract important text that matches patterns you describe." 
df.loc["regular_expressions_groups", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Define a pattern in regex notation that uses a capturing group&- Define a pattern in regex notation that uses the `|` symbol as a logical +Or+ &- Define a pattern in regex notation that uses a non-capturing group&&" 
df.loc["regular_expressions_groups", "Prerequisties"] = "Learners should have some experience composing and using simple regular expressions (regex).  For an introduction to regular expression concepts and using regular expression checkers like [Regex101](https://www.regex101.com/), consider the [Demystifying Regular Expressions](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_regular_expressions/demystifying_regular_expressions.md#1) module.  To learn how to compose and use simple regular expressions, consider the [Regular Expressions Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_basics/regular_expressions_basics.md#1) module.&" 
df.loc["regular_expressions_groups", "Sets You Up For"] = "  regular_expressions_anchors_boundaries " 
df.loc["regular_expressions_groups", "Depends On Knowledge In"] = "  demystifying_regular_expressions   regular_expressions_basics " 
df.loc["regular_expressions_lookaheads", "title"] = "Regular Expressions: Lookaheads"
df.loc["regular_expressions_lookaheads", "author"] = "Joy Payton"
df.loc["regular_expressions_lookaheads", "estimated_time_in_minutes"] = "30"
df.loc["regular_expressions_lookaheads", "good_first_module"] = "false" 
df.loc["regular_expressions_lookaheads", "coding_required"] = "true"
df.loc["regular_expressions_lookaheads", "coding_level"] = "intermediate"
df.loc["regular_expressions_lookaheads", "sequence_name"] = "regex"
df.loc["regular_expressions_lookaheads", "comment"] = "Use regular expressions, or regex, for complex pattern matching involving lookaheads." 
df.loc["regular_expressions_lookaheads", "long_description"] = "Learn to compose intermediate regular expressions using lookahead syntax, in order to identify text that matches patterns you describe." 
df.loc["regular_expressions_lookaheads", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Explain the difference between +moving+ ahead and +looking+ ahead in regular expression parsing&- Explain why a +lookahead+ can be useful in a regular expression&&" 
df.loc["regular_expressions_lookaheads", "Prerequisties"] = "Learners should have some experience composing and using simple regular expressions (regex), including the use of capturing groups.  For an introduction to regular expression concepts, consider the [Demystifying Regular Expressions](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_regular_expressions/demystifying_regular_expressions.md#1) module.  To learn how to compose and use simple regular expressions, consider the [Regular Expressions Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_basics/regular_expressions_basics.md#1) module. The [Regular Expressions: Groups](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_groups/regular_expressions_groups.md#1) module will introduce you to capturing and non-capturing groups if those are new to you or you would like a refresher.&" 
df.loc["regular_expressions_lookaheads", "Sets You Up For"] = " " 
df.loc["regular_expressions_lookaheads", "Depends On Knowledge In"] = "  demystifying_regular_expressions   regular_expressions_basics   regular_expressions_groups " 
df.loc["reproducibility", "title"] = "Reproducibility, Generalizability, and Reuse"
df.loc["reproducibility", "author"] = "Joy Payton"
df.loc["reproducibility", "estimated_time_in_minutes"] = "60"
df.loc["reproducibility", "good_first_module"] = "true" 
df.loc["reproducibility", "coding_required"] = "false"
df.loc["reproducibility", "comment"] = "This module provides learners with an approachable introduction to the concepts and impact of **research reproducibility**, **generalizability**, and **data reuse**, and how technical approaches can help make these goals more attainable." 
df.loc["reproducibility", "long_description"] = "**If you currently conduct research or expect to in the future**, the concepts we talk about here are important to grasp.  This material will help you understand much of the current literature and debate around how research should be conducted, and will provide you with a starting point for understanding why some practices (like writing code, even for researchers who have never programmed a computer) are gaining traction in the research field.  **If research doesn't form part of your future plans, but you want to *use* research** (for example, as a clinician or public health official), this material will help you form criteria for what research to consider the most rigorous and useful and help you understand why science can seem to vacillate or be self-contradictory." 
df.loc["reproducibility", "learning_objectives"] = "&After completion of this module, learners will be able to:&&* Explain the benefits of conducting research that is **reproducible** (can be re-done by a different, unaffiliated scientist)&* Describe how technological approaches can help research be more reproducible&* Argue in support of practices that organize and describe documents, datasets, and other files as a way to make research more reproducible&&" 
df.loc["reproducibility", "Prerequisties"] = "&It is helpful if learners have conducted research, are familiar with -- by reading or writing -- peer-reviewed literature, and have experience using data and methods developed by other people.  There is no need to have any specific scientific or medical domain knowledge or technical background.    &" 
df.loc["sql_basics", "title"] = "SQL Basics"
df.loc["sql_basics", "author"] = "Peter Camacho"
df.loc["sql_basics", "estimated_time_in_minutes"] = ""
df.loc["sql_basics", "comment"] = "Structured Query Language, or SQL, is a relational database solution that has been around for decades.  Learn how to do basic SQL queries on single tables, by using code, hands-on." 
df.loc["sql_basics", "long_description"] = "Do you want to learn basic Structured Query Language (SQL) either to understand concepts or prepare for access to a relational database?  This module will give you hands on experience with simple queries using keywords including SELECT, WHERE, FROM, DISTINCT, and AS.  We'll also briefly cover working with empty (NULL) values using IS NULL and IS NOT NULL.  This module is appropriate for people who have little or no experience in SQL and are ready to practice with real queries." 
df.loc["sql_basics", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Use SELECT, FROM, and WHERE to do a basic query on a SQL table&- Use IS NULL and IS NOT NULL operators to work with empty values&- Explain the use of DISTINCT and how it can be useful&- Use AS and ORDER BY to change how query results appear&- Explain why the LIMIT keyword can be useful&&&" 
df.loc["sql_intermediate", "title"] = "SQL, Intermediate Level"
df.loc["sql_intermediate", "author"] = "Peter Camacho; Joy Payton"
df.loc["sql_intermediate", "estimated_time_in_minutes"] = "60"
df.loc["sql_intermediate", "good_first_module"] = "false" 
df.loc["sql_intermediate", "coding_required"] = "true"
df.loc["sql_intermediate", "coding_language"] = "sql"
df.loc["sql_intermediate", "coding_level"] = "basic"
df.loc["sql_intermediate", "sequence_name"] = "sql"
df.loc["sql_intermediate", "data_task"] = "data_wrangling"
df.loc["sql_intermediate", "comment"] = "Learn how to do intermediate SQL queries on single tables, by using code, hands-on." 
df.loc["sql_intermediate", "long_description"] = "Do you want to learn intermediate Structured Query Language (SQL) for more precise and complex data querying on single tables?  This module will give you hands on experience with single-table queries using keywords including CASE, LIKE, REGEXP_LIKE, GROUP BY, HAVING, and WITH, along with a number of aggregate functions like COUNT and AVG.  This module is appropriate for people who are comfortable writing basic SQL queries and are ready to practice more advanced sklls." 
df.loc["sql_intermediate", "learning_objectives"] = "&After completion of this module, learners will be able to:&&* Create new data classifications using `CASE` statements&* Find text that matches a given pattern using `LIKE` and `REGEXP_LIKE` statements&* Use `GROUP BY` and `HAVING` statements along with aggregate functions to understand group characteristics&* Use `WITH` to create sub queries&&" 
df.loc["sql_intermediate", "Prerequisties"] = "&Some experience writing basic SQL code (SELECT, FROM, WHERE) is expected in this module.  If you would like a code-free overview to SQL we recommend our module [Demystifying SQL](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/demystifying_sql/demystifying_sql.md).  If you need to develop basic SQL fluency we recommend our module [SQL Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/sql_basics/sql_basics.md).&&" 
df.loc["sql_intermediate", "Sets You Up For"] = "   sql_joins  " 
df.loc["sql_intermediate", "Depends On Knowledge In"] = "   demystifying_sql   sql_basics  " 
df.loc["sql_joins", "title"] = "SQL Joins"
df.loc["sql_joins", "author"] = "Joy Payton"
df.loc["sql_joins", "estimated_time_in_minutes"] = ""
df.loc["sql_joins", "comment"] = "Learn about SQL joins: what they accomplish, and how to write them." 
df.loc["sql_joins", "long_description"] = "Usually, data in a SQL database is organized into multiple interrelated tables.  This means you will often have to bring data together from two or more tables into a single dataset to answer your research questions.  This +join+ action is accomplished using `JOIN` commands.  This module teaches types of joins, join criteria, and how to write `JOIN` code." 
df.loc["sql_joins", "learning_objectives"] = "After completion of this module, learners will be able to:&&- Understand the parts of a JOIN&- Describe the +shapes+ of SQL JOINs: inner, left, right, and full&- Explain what +join criteria+ are&&" 
df.loc["statistical_tests", "title"] = "Statistical Tests in Open Source Software"
df.loc["statistical_tests", "author"] = "Rose Hartman"
df.loc["statistical_tests", "estimated_time_in_minutes"] = ""
df.loc["statistical_tests", "comment"] = "This module provides an overview of the most commonly used kinds of statistical tests and links to code for running many of them in both R and python." 
df.loc["statistical_tests", "long_description"] = "This module contains a curated list of links to tutorials and examples of many common statistical tests in both R and python. If you want to use R or python for data analysis but aren't sure how to write code for the statistical tests you want to run, this is a great place to start. This will be an especially valuable resource for people who have experience conducting analysis in other software (e.g. SAS, SPSS, MPlus, Matlab) and are looking to move to R and/or python. If you are new to data analysis, this module provides some structure to help you think about which statistical tests to run, and examples of code to execute them. It doesn't cover the statistical theory itself, though, so you'll need to do some additional reading before applying the code for any tests you don't already understand (there are recommended resources for learning statistical techniques at the end of the module)." 
df.loc["statistical_tests", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Use four key questions to help determine which statistical tests will be most appropriate in a given situation&- Discuss general differences between running statistical tests in R vs. python&- Quickly find the code they need to be able to run most common statistical tests in R or python&&" 
df.loc["tidy_data", "title"] = "Tidy Data"
df.loc["tidy_data", "author"] = "Joy Payton"
df.loc["tidy_data", "estimated_time_in_minutes"] = ""
df.loc["tidy_data", "comment"] = "Tidy is a technical term in data analysis and describes an optimal way for organizing data that will be analyzed computationally." 
df.loc["tidy_data", "long_description"] = "Are you concerned about how to organize your data so that it's easier to work with in a computational solution like R, Python, or other statistical software?  This module will explain the concept of +tidy data+, which will help make analysis and data reuse a bit simpler." 
df.loc["tidy_data", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Describe the three characteristics of tidy data&- Describe how messy data could be transformed into tidy data&- Describe the three tenets of tidy analysis&&&" 
df.loc["using_redcap_api", "title"] = "Using the REDCap API"
df.loc["using_redcap_api", "author"] = "Joy Payton"
df.loc["using_redcap_api", "estimated_time_in_minutes"] = ""
df.loc["using_redcap_api", "comment"] = "REDCap is a research data capture tool used by many researchers in basic, translational, and clinical research efforts.  Learn how to use the REDCap API in this module." 
df.loc["using_redcap_api", "long_description"] = "If your institution provides access to REDCap, this module is right for you.  REDCap is a convenient and powerful way to collect and store research data.  This module will teach you how to interact with the REDCap API, or +Application Programming Interface,+ which can help you automate your data analysis. This will also help you understand APIs in general and what makes their use so appealing for reproducible research efforts." 
df.loc["using_redcap_api", "learning_objectives"] = "&After completion of this module, learners will be able to:&&- Define what an API is and why it's useful to researchers&- Enable API usage on REDCap projects&- Use the REDCap API to pull data into an R or Python data analysis&&&" 
df["Linked Courses"] = [list() for x in range(len(df.index))]
a = df.loc["bash_103_combining_commands", "Linked Courses"]
a.append("bash_command_line_101")
a.append("bash_command_line_102")
a.append("directories_and_file_paths")
df.at["bash_103_combining_commands", "Linked Courses"] = list(a)
a = df.loc["bash_command_line_101", "Linked Courses"]
a.append("directories_and_file_paths")
a.append("git_setup_windows")
df.at["bash_command_line_101", "Linked Courses"] = list(a)
a = df.loc["bash_command_line_102", "Linked Courses"]
a.append("bash_103_combining_commands")
a.append("bash_command_line_101")
a.append("bash_conditionals_loops")
a.append("directories_and_file_paths")
df.at["bash_command_line_102", "Linked Courses"] = list(a)
a = df.loc["bash_conditionals_loops", "Linked Courses"]
a.append("bash_103_combining_commands")
a.append("bash_command_line_101")
a.append("bash_command_line_102")
a.append("bash_scripts")
a.append("directories_and_file_paths")
df.at["bash_conditionals_loops", "Linked Courses"] = list(a)
a = df.loc["bash_scripts", "Linked Courses"]
a.append("bash_command_line_101")
a.append("reproducibility")
df.at["bash_scripts", "Linked Courses"] = list(a)
a = df.loc["bias_variance_tradeoff", "Linked Courses"]
a.append("demystifying_machine_learning")
df.at["bias_variance_tradeoff", "Linked Courses"] = list(a)
a = df.loc["citizen_science", "Linked Courses"]
df.at["citizen_science", "Linked Courses"] = list(a)
a = df.loc["data_management_basics", "Linked Courses"]
a.append("reproducibility")
df.at["data_management_basics", "Linked Courses"] = list(a)
a = df.loc["data_storage_models", "Linked Courses"]
df.at["data_storage_models", "Linked Courses"] = list(a)
a = df.loc["data_visualization_in_ggplot2", "Linked Courses"]
a.append("data_visualization_in_open_source_software")
a.append("data_visualization_in_seaborn")
a.append("r_basics_introduction")
a.append("r_practice")
a.append("statistical_tests")
df.at["data_visualization_in_ggplot2", "Linked Courses"] = list(a)
a = df.loc["data_visualization_in_open_source_software", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("data_visualization_in_seaborn")
df.at["data_visualization_in_open_source_software", "Linked Courses"] = list(a)
a = df.loc["data_visualization_in_seaborn", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("data_visualization_in_open_source_software")
a.append("demystifying_python")
a.append("python_practice")
a.append("statistical_tests")
df.at["data_visualization_in_seaborn", "Linked Courses"] = list(a)
a = df.loc["database_normalization", "Linked Courses"]
df.at["database_normalization", "Linked Courses"] = list(a)
a = df.loc["demystifying_containers", "Linked Courses"]
a.append("docker_101")
a.append("reproducibility")
df.at["demystifying_containers", "Linked Courses"] = list(a)
a = df.loc["demystifying_geospatial_data", "Linked Courses"]
a.append("geocode_lat_long")
df.at["demystifying_geospatial_data", "Linked Courses"] = list(a)
a = df.loc["demystifying_large_language_models", "Linked Courses"]
df.at["demystifying_large_language_models", "Linked Courses"] = list(a)
a = df.loc["demystifying_machine_learning", "Linked Courses"]
a.append("bias_variance_tradeoff")
df.at["demystifying_machine_learning", "Linked Courses"] = list(a)
a = df.loc["demystifying_python", "Linked Courses"]
a.append("bash_command_line_101")
a.append("python_basics_variables_functions_methods")
df.at["demystifying_python", "Linked Courses"] = list(a)
a = df.loc["demystifying_regular_expressions", "Linked Courses"]
a.append("regular_expressions_basics")
df.at["demystifying_regular_expressions", "Linked Courses"] = list(a)
a = df.loc["demystifying_sql", "Linked Courses"]
a.append("reproducibility")
df.at["demystifying_sql", "Linked Courses"] = list(a)
a = df.loc["directories_and_file_paths", "Linked Courses"]
df.at["directories_and_file_paths", "Linked Courses"] = list(a)
a = df.loc["docker_101", "Linked Courses"]
a.append("demystifying_containers")
df.at["docker_101", "Linked Courses"] = list(a)
a = df.loc["elements_of_maps", "Linked Courses"]
df.at["elements_of_maps", "Linked Courses"] = list(a)
a = df.loc["genomics_quality_control", "Linked Courses"]
a.append("bash_103_combining_commands")
a.append("bash_command_line_101")
a.append("bash_command_line_102")
a.append("bash_conditionals_loops")
a.append("data_storage_models")
a.append("directories_and_file_paths")
a.append("genomics_setup")
a.append("omics_orientation")
df.at["genomics_quality_control", "Linked Courses"] = list(a)
a = df.loc["genomics_setup", "Linked Courses"]
a.append("bash_command_line_101")
df.at["genomics_setup", "Linked Courses"] = list(a)
a = df.loc["geocode_lat_long", "Linked Courses"]
df.at["geocode_lat_long", "Linked Courses"] = list(a)
a = df.loc["git_creation_and_tracking", "Linked Courses"]
a.append("git_setup_mac_and_linux")
a.append("git_setup_windows")
df.at["git_creation_and_tracking", "Linked Courses"] = list(a)
a = df.loc["git_history_of_project", "Linked Courses"]
a.append("bash_command_line_101")
a.append("git_creation_and_tracking")
a.append("git_intro")
a.append("git_setup_mac_and_linux")
a.append("git_setup_windows")
df.at["git_history_of_project", "Linked Courses"] = list(a)
a = df.loc["git_intro", "Linked Courses"]
df.at["git_intro", "Linked Courses"] = list(a)
a = df.loc["git_setup_mac_and_linux", "Linked Courses"]
a.append("git_setup_windows")
df.at["git_setup_mac_and_linux", "Linked Courses"] = list(a)
a = df.loc["git_setup_windows", "Linked Courses"]
a.append("git_setup_mac_and_linux")
df.at["git_setup_windows", "Linked Courses"] = list(a)
a = df.loc["how_to_troubleshoot", "Linked Courses"]
df.at["how_to_troubleshoot", "Linked Courses"] = list(a)
a = df.loc["intro_to_nhst", "Linked Courses"]
a.append("statistical_tests")
df.at["intro_to_nhst", "Linked Courses"] = list(a)
a = df.loc["learning_to_learn", "Linked Courses"]
a.append("reproducibility")
df.at["learning_to_learn", "Linked Courses"] = list(a)
a = df.loc["omics_orientation", "Linked Courses"]
df.at["omics_orientation", "Linked Courses"] = list(a)
a = df.loc["pandas_transform", "Linked Courses"]
a.append("python_basics_lists_dictionaries")
a.append("python_basics_loops_conditionals")
a.append("python_basics_variables_functions_methods")
a.append("python_practice")
df.at["pandas_transform", "Linked Courses"] = list(a)
a = df.loc["python_basics_exercise", "Linked Courses"]
a.append("demystifying_python")
a.append("python_basics_lists_dictionaries")
a.append("python_basics_loops_conditionals")
a.append("python_basics_variables_functions_methods")
df.at["python_basics_exercise", "Linked Courses"] = list(a)
a = df.loc["python_basics_lists_dictionaries", "Linked Courses"]
a.append("demystifying_python")
a.append("pandas_transform")
a.append("python_basics_exercise")
a.append("python_basics_loops_conditionals")
a.append("python_basics_variables_functions_methods")
df.at["python_basics_lists_dictionaries", "Linked Courses"] = list(a)
a = df.loc["python_basics_loops_conditionals", "Linked Courses"]
a.append("demystifying_python")
a.append("pandas_transform")
a.append("python_basics_exercise")
a.append("python_basics_lists_dictionaries")
a.append("python_basics_variables_functions_methods")
df.at["python_basics_loops_conditionals", "Linked Courses"] = list(a)
a = df.loc["python_basics_variables_functions_methods", "Linked Courses"]
a.append("demystifying_python")
a.append("python_basics_exercise")
a.append("python_basics_loops_conditionals")
df.at["python_basics_variables_functions_methods", "Linked Courses"] = list(a)
a = df.loc["python_practice", "Linked Courses"]
a.append("data_visualization_in_seaborn")
a.append("demystifying_python")
a.append("pandas_transform")
a.append("python_basics_lists_dictionaries")
a.append("python_basics_loops_conditionals")
a.append("python_basics_variables_functions_methods")
a.append("r_practice")
df.at["python_practice", "Linked Courses"] = list(a)
a = df.loc["r_basics_introduction", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("r_basics_transform_data")
a.append("r_basics_visualize_data")
a.append("r_missing_values")
a.append("r_practice")
a.append("r_summary_stats")
a.append("reproducibility")
df.at["r_basics_introduction", "Linked Courses"] = list(a)
a = df.loc["r_basics_transform_data", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("r_basics_introduction")
a.append("r_basics_visualize_data")
a.append("r_missing_values")
a.append("r_practice")
a.append("r_reshape_long_wide")
a.append("r_summary_stats")
df.at["r_basics_transform_data", "Linked Courses"] = list(a)
a = df.loc["r_basics_visualize_data", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("r_basics_introduction")
a.append("r_practice")
a.append("tidy_data")
df.at["r_basics_visualize_data", "Linked Courses"] = list(a)
a = df.loc["r_missing_values", "Linked Courses"]
a.append("r_basics_introduction")
a.append("r_basics_transform_data")
a.append("r_practice")
df.at["r_missing_values", "Linked Courses"] = list(a)
a = df.loc["r_practice", "Linked Courses"]
a.append("data_visualization_in_ggplot2")
a.append("python_practice")
a.append("r_basics_introduction")
a.append("r_basics_transform_data")
df.at["r_practice", "Linked Courses"] = list(a)
a = df.loc["r_reshape_long_wide", "Linked Courses"]
a.append("r_basics_introduction")
a.append("r_basics_transform_data")
a.append("r_practice")
a.append("tidy_data")
df.at["r_reshape_long_wide", "Linked Courses"] = list(a)
a = df.loc["r_summary_stats", "Linked Courses"]
a.append("directories_and_file_paths")
a.append("r_basics_introduction")
a.append("r_basics_transform_data")
df.at["r_summary_stats", "Linked Courses"] = list(a)
a = df.loc["regular_expressions_basics", "Linked Courses"]
a.append("demystifying_regular_expressions")
df.at["regular_expressions_basics", "Linked Courses"] = list(a)
a = df.loc["regular_expressions_boundaries_anchors", "Linked Courses"]
a.append("demystifying_regular_expressions")
a.append("regular_expressions_basics")
a.append("regular_expressions_groups")
df.at["regular_expressions_boundaries_anchors", "Linked Courses"] = list(a)
a = df.loc["regular_expressions_groups", "Linked Courses"]
a.append("demystifying_regular_expressions")
a.append("regular_expressions_basics")
df.at["regular_expressions_groups", "Linked Courses"] = list(a)
a = df.loc["regular_expressions_lookaheads", "Linked Courses"]
a.append("demystifying_regular_expressions")
a.append("regular_expressions_basics")
a.append("regular_expressions_boundaries_anchors")
a.append("regular_expressions_groups")
df.at["regular_expressions_lookaheads", "Linked Courses"] = list(a)
a = df.loc["reproducibility", "Linked Courses"]
a.append("git_intro")
df.at["reproducibility", "Linked Courses"] = list(a)
a = df.loc["sql_basics", "Linked Courses"]
a.append("demystifying_sql")
df.at["sql_basics", "Linked Courses"] = list(a)
a = df.loc["sql_intermediate", "Linked Courses"]
a.append("demystifying_regular_expressions")
a.append("demystifying_sql")
a.append("regular_expressions_basics")
a.append("sql_basics")
a.append("sql_joins")
df.at["sql_intermediate", "Linked Courses"] = list(a)
a = df.loc["sql_joins", "Linked Courses"]
a.append("database_normalization")
a.append("sql_basics")
a.append("sql_intermediate")
df.at["sql_joins", "Linked Courses"] = list(a)
a = df.loc["statistical_tests", "Linked Courses"]
a.append("data_visualization_in_open_source_software")
a.append("python_basics_variables_functions_methods")
a.append("r_basics_introduction")
df.at["statistical_tests", "Linked Courses"] = list(a)
a = df.loc["tidy_data", "Linked Courses"]
a.append("reproducibility")
df.at["tidy_data", "Linked Courses"] = list(a)
a = df.loc["using_redcap_api", "Linked Courses"]
a.append("bash_command_line_101")
a.append("git_creation_and_tracking")
a.append("reproducibility")
df.at["using_redcap_api", "Linked Courses"] = list(a)
