<!--

author:   Joy Payton
email:    paytonk@chop.edu
version:  1.0.0
current_version_description: Initial version
module_type: standard
docs_version: 1.1.0
language: en
narrator: US English Female
mode: Textbook

title: Regular Expressions in SQL

comment:  Learn to use regular expressions in SQL

long_description: Ready to use regular expressions (regex) in SQL?  This module will help you put what you know about regular expressions into action in SQL.

estimated_time_in_minutes: 30

@pre_reqs
Some experience writing basic SQL code (SELECT, FROM, WHERE) is expected in this module.  If you need to develop basic SQL fluency we recommend our module [SQL Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/sql_basics/sql_basics.md).  Additionally, some experience using regular expressions is necessary.  To learn how to compose and use simple regular expressions, consider the [Regular Expressions Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_basics/regular_expressions_basics.md#1) module.  More advanced material can be found in [Regular Expressions: Intermediate](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_intermediate/regular_expressions_intermediate.md#1)
@end

@learning_objectives  
After completion of this module, learners will be able to:

- identify key elements
- create a product
- do a task
- articulate the rationale for something
@end

@version_history 

Previous versions: 

- [x.x.x](link): that version's current version description
- [x.x.x](link): that version's current version description
- [x.x.x](link): that version's current version description
@end

import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros.md
import: https://raw.githubusercontent.com/arcus/education_modules/main/_module_templates/macros_sql.md 
-->

# Module Title

@overview

## A Brief Refresher

These concepts should seem familiar to you.  If this brief refresher seems to be new information, consider reviewing [SQL Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/sql_basics/sql_basics.md) or [Regular Expressions Basics](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/regular_expressions_basics/regular_expressions_basics.md#1).

**SQL** (**S**tructured **Q**uery **L**anguage) is a language that for more than four decades has been used to interact with **relational databases**, which store rectangular data in rows and columns (also known as fields).  A common pattern for SQL queries is `SELECT ... FROM ... WHERE`.

A **regular expression** (also known as **regex**) is a specific way to express a rule for a pattern, such as the pattern you expect for an email address, or the pattern for a credit card number.  An example regular expression is `\d{4}`, which describes a four digit number, such as year. 

These two technologies can intersect, and this module will go into how to use regular expressions in your SQL queries.

## The Intersection of SQL and Regex

Why would you use regular expressions in SQL?  Well, perhaps you want to `SELECT`  email addresses that end in ".edu", or want to exclude from your search lab values that have non-numeric characters such as `>` or `+`.  Being able to use patterns in your SQL code will allow you to be more precise in the query results you obtain.

In order to work with regular expressions in SQL, you need to know how to work with:

* Regular expression operators in SQL (`REGEXP`, `NOT REGEXP`)
* Special regular expression syntax in SQL
* Posix operators

We'll cover each of these topics in the next few sections.

### `REGEXP` Operator

We'll start with a regular expression operator in SQL which will be very useful for you: `REGEXP`.

`REGEXP` compares a pattern to a text and will return `True` if the text matches the pattern and `False` if the text does not match the pattern.  This means you can use it as part of a `WHERE` clause to return only the rows where a certain pattern appears in a given column.  If the pattern is matched, the row will be returned, and if the pattern isn't matched, the row won't be returned in your result.

Let's consider some fabricated data including data on patients, their allergies, and a few laboratory observations. 

Take a quick peek at the `allergies` table by running the SQL command below (by clicking on the "play" / "run" button beneath the SQL code).

```sql
SELECT *
FROM allergies
LIMIT 10;
```
@AlaSQL.eval("#dataTable4a")

<table id="dataTable4a" border="1"></table><br>

<div style = "display:none;">

</div>

We have visually scanned through some of the data but we want to make sure we understand the kinds of nut allergies that are included.  Are tree nuts specified?  Peanut is certainly included.  What about coconuts?  Are there specific nut types with "nut" in their name also included, like walnut or macadamia nut?  Let's get all the distinct allergy types that include the string "nut" or "nuts" at the end of a word.  

We'll start our regular expression (regex) with `.*`, indicating that our pattern starts with zero to many characters of any kind.  We then follow that with `nut`, which indicates that literal text, and we end our pattern with another `.*`, indicating that our pattern ends with zero to many characters of any kind.  

```sql
SELECT DISTINCT description
FROM allergies
WHERE description REGEXP '.*nut.*';
```
@AlaSQL.eval("#dataTable4b")

<table id="dataTable4b" border="1"></table><br>

Want to try this yourself?  Try determining what kinds of "fish" allergies there are, and what kinds of "pollen" allergies there are by altering the regular expression in the query above.

### Negating `REGEXP`

Take a quick peek at the `observations` table.

```sql
SELECT *
FROM observations
LIMIT 10;
```
@AlaSQL.eval("#dataTable5a")

<table id="dataTable5a" border="1"></table><br>

In this limited output from the `observations` table, take a look at the time stamps in the first column, `observed`.

Can we dependably state that the pattern of our time stamp is four digit year, two digit month, and two digit day, with date fields separated by hyphens, followed by the capital letter T, and a 24 hour clock time in the format HH:MM:SS followed by the letter Z?  It seems like that's the case, but can we be certain?  We've been asked to confirm this by a contractor who is updating our clinical systems.  

We can construct a pattern to check against:

* `\d{4}-\d{2}-\d{2}` will represent our date information (we can use just the hyphen by itself, since it's not enclosed in square brackets, or we can be extra safe and add a backslash before the hyphen)
* `T\d{2}:\d{2}:\d{2}Z` will represent our time information.

This time, we want to get all the data results where the pattern does **not** match, so we'll add the NOT qualifier to our where clause:

```sql
SELECT *
FROM observations
WHERE NOT observed REGEXP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z';
```
@AlaSQL.eval("#dataTable5b")

<table id="dataTable5b" border="1"></table><br>

OK, whew, it looks like there's no data where the pattern doesn't match.  That message might make you feel uneasy.  How can you know it's not returning data because there's no match, and it's not the case instead that it's not returning data because the query is badly written?  

This is a great instinct to have -- to question the results of a query.  Think about how you might change your query to see if you have the syntax right.  

There are lots of ways to check:

* Try leaving out the `NOT`, and make sure all the pattern matches come through.  You should have a result set that is the same size as the table.
* Or, you could try changing the pattern to one you know should fail for some or all of the rows.  For example, change the trailing `Z` to `Q`.  Again, if we have our pattern right, and it matches each row, changing that final letter should mean every row now fails to match the new (incorrect) pattern.

<div class = "important">
<b style="color: rgb(var(--color-highlight));">Important note</b><br>

AlaSQL is the flavor (or dialect) of SQL we're using to practice here, because it works well in a browser.  However, the SQL flavor for your databases might have some differences.  For example, you  might be able to use `REGEXP NOT` as the negated regular expression check, instead of using `NOT ... REGEXP` as we've done here.

Small syntax differences like these are a good reason why it can be helpful to practice some healthy skepticism and check your SQL work by changing around the query to intentionally get different results.

[![LiaScript](https://raw.githubusercontent.com/LiaScript/LiaScript/master/badges/course.svg)](https://liascript.github.io/course/?https://raw.githubusercontent.com/arcus/education_modules/main/docs.md)

</div>

<div style = "display:none;">

@AlaSQL.buildTable_patients
@AlaSQL.buildTable_allergies
@AlaSQL.buildTable_observations

</div>

### `REGEXP_LIKE` Operator

How is `REGEXP` different from `REGEXP_LIKE`, a SQL operator you may have had prior exposure to?

The short answer is that they are synonyms or aliases.  You can use `REGEXP` and `REGEXP_LIKE` in much the same way.  They do have slightly different structures, however, in the way you write them.  Execute the two queries in the next two chunks of SQL code to see how these two forms accomplish the same thing.  The pattern we're using is identifying addresses with apartment or unit numbers, to help us find a cohort of patients who are likely to be renters.

```sql
SELECT *
FROM patients
WHERE LOWER(address) REGEXP ' apt\.? ';
```
@AlaSQL.eval("#dataTable5b")

<table id="dataTable5b" border="1"></table><br>




* In MySQL, `REGEXP`, `RLIKE`, and `REGEXP_LIKE()` all accomplish the same thing.




<div style = "display:none;">

@AlaSQL.buildTable_patients
@AlaSQL.buildTable_allergies
@AlaSQL.buildTable_observations

</div>

## Feedback

@feedback
