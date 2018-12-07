# Deployment Experience

I'm taking notes when deploy machine learning model on production.

## Golden Rule
* Even if you did most of the work, remember to praise your teammates for their support and their efforts. Say that repeatly. As for what to say to your manager, it depends. But always remember to appreciate their time and patience for any question you are not certain to answer to production team.

## Preparation
### From Data Science Team
#### Feature Set
* Make sure production team will understand the data source, and how to generate the features.
* There might be system limitation, that they will tell you which feature cannot be generated on production platform. Make sure you can understand the reasons behind. In the future, you can also avoid to create those things that production team definitely won't implement.
  * If possible, also try to see whether there is any alternative, so that you don't need to simply remove some important features.
#### Model Artifact
* Data Science Team trained the model, production team can just call the trained model. In this way, it might be possible to achieve real time execution. 
  * For Example, you are using python or R to create the models, production team is using other languages. If their language can load the model (this might be .pickle file, binary file or any other file), it's good. You just need to save your model to the right format they need.
* Relevant param settings, configurations should also provide to production team.
* Configurable Settings - Make sure production team is able to make the model, feature generation flexible enough for different clients' settings
#### Feature Preprocessing
* You might also need to tell production team how did you preprocess the features, to see whether they can do the same thing.

### From Architect
* The Architect will prepare the system structure which indicates external libraries/sources, components that need to add, components already exist and any other changes can be done but not the priority.
* There might be also some data filtering criteria they will require, so that you need to filter them out before the data hit the model.
* They will also make assumptions, so that your feature generation, model creation need to satisfy these assumptions.

