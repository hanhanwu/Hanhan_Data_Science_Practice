'''
Created on Apr 28, 2018

This module includes the child classes that inherent the summer_dessert abstract class
'''
from summer_dessert import SummerDessert

class Icecream(SummerDessert):
    """
    ice-cream in Totoro's summer dessert shop
    """
    # Here's how we add more variables in __init__() besides those defined in the super class.
    def __init__(self, has_waffle, has_bubble, flavor_count, discount_price):
    # Get variables from SummerDessert class
    super(Icecream, self).__init__(has_waffle, has_bubble, flavor_count, discount_price)
    
    self.mochi_count = 9
    
    # Here it overwrites the value assigned in the super class and will affect the function definied in the super class
    base_unit_price = 7 
    
    def dessert_type(self):
        print 'ice-cream!'
        
    def count_mochi(self):
        print(self.mochi_count)
        
        
class Galeto(SummerDessert):
    """
    galeto in Totoro's summer dessert shop
    """
    
    base_unit_price = 9
    
    def dessert_type(self):
        print 'galeto!'
        
        
class FrozenYogurt(SummerDessert):
    """
    frozen yogurt in Totoro's summer dessert shop
    """
    
    base_unit_price = 5
    
    def dessert_type(self):
        print 'frozen yogurt!'
