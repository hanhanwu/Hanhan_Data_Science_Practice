When you are going to build the whole system for your data science pipeline, gradually you will think about OO design to make your code re-usable and more organized. It's been a while that I haven't thought about OO in my coding and haven't thought about design patterns. Time to do a review and some practice.

*************************************************************************************

BASIC SUMMER DESSERTS OO DESIGN REVIEW

When I say "summer desserts", it's not any fancy pyton module, or anything like android OS, it's just summer desserts, an abstract class which can be inherented by ice-cream, galeto, frozen yogurt, etc.

* My code: 
  * [summer_dessert.py][1] contains the abstract class that inherented by specific summer desserts
    * static variables belongs to the whole class
    * abstractmethod needs python `abc` class, it can be empty in the abstract class, so that later the descendent class can do specific implementation
  * [summer_dessert_children.py][2] contains the descendent classes, they all inherent from the abstract class `SummerDessert`, and implemented their own `dessert_type()`, which was an abstract method in the abstract class
  * [customer.py][3] creats the specific class instances

* Relevant Resources
  * Python `abc` class, from where you can get `abstractclass`: https://www.python-course.eu/python3_abstract_classes.php
  * abstract class, abstract method tutorial: https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/
  
  
[1]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Python_OO_Design/basic_design_summer_dessert/summer_dessert.py  
[2]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Python_OO_Design/basic_design_summer_dessert/summer_dessert_children.py
[3]:https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/Python_OO_Design/basic_design_summer_dessert/customer.py
