<h1 align='center'> Assignment 3: Project <br>Analyzing the French Journal Officiel </h1>
<p align='center'>
<i>CentraleSupélec <br>
2017 - 2018 <hr></i></p>

__Authors__: Adib Baziz, Reuben Dorent, Samuel Joutard, Joël Seytre<br><br>
__Teacher__: [Fragkiskos D. Malliaros](http://fragkiskos.me/)

## Index
1. [Content](#content)
2. [Acknowledgements](#thanks)
3. [Setting up](#setup)

# <a name="content"></a>Content
* **project-proposal-ngsa.pdf** is the formal project proposal submitted by the team.
* **previous-project-report.pdf** is the report of the project previously worked on by Alexis Thual and Joël Seytre for the MVA course _Graphs in Machine Learning_ taught by Michal Valko.

# <a name="thanks"></a>Acknowledgements
Special thanks to Alexis Thual for his work on parsing the French Journal Officiel.<br>
The [parsing-journal-officiel](parsing-journal-officiel-master) folder is a copy of his [github](https://github.com/alexis-thual/parsing-journal-officiel) as of late March 2018.

# <a name="thanks"></a>Setting up
* Installation guide to ElasticSearch [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html). <br>
If necessary, full guide to ElasticSearch [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html).<br>
Remember that you will need an updated JDK (> Java 8, Oracle JDK version 1.8.0_131 as of today).<br>
If you want to, choosing installing ES **not** as a service and without any optional package works just fine.<br>
_Note: if you encounter some JNA problems when trying to run ElasticSearch, make sure to grant security privileges to your ElasticSearch folder_
* Follow the instructions of our local copy of Alexis Thual's [parsing-journal-officiel](parsing-journal-officiel-master).<br>
Expect it to last several hours.<br>
**Summary:**
    * `cd ElasticSearch/xxx/bin`
    * `.\elasticsearch.exe`
    * in another terminal `cd parsing-journal-officiel/parsing_xml`
    * `python3 .\fullDeploy.py`

