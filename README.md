# The application :

This project is a flask application which allows to recommend Stack Overflow tags to any given message.
It uses different LDA and Word2Vec models.
This models have been trained thanks to a querry built on
https://data.stackexchange.com/stackoverflow/query/new . The corresponding query was :

```
      SELECT * FROM posts WHERE Id < 50000
```
Because of the time factor (computer science is changing rapidly), it is better to test the application on old messages. For example, testing it on messages providing from the querry:

```
      SELECT * FROM posts WHERE Id BETWEEN 50000 AND 50500
```

## How to use it

### First steps

Here are the instructions which allows to launch the application :
![alt text](https://github.com/E-tanok/NLTK_stackoverflow_tags_recommender/blob/master/project_instructions/first_steps.png)


### Once the application is launched

Just write (or copy/paste) the message of your choice :
![alt text](https://github.com/E-tanok/NLTK_stackoverflow_tags_recommender/blob/master/project_instructions/writing_message.png)


### The recommendation results

The recommendation algorithm generates tags for you :
![alt text](https://github.com/E-tanok/NLTK_stackoverflow_tags_recommender/blob/master/project_instructions/results.png)
