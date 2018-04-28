'''
Created on Apr 28, 2018

This module includes the child classes that inherent the summer_dessert abstract class
'''
from summer_dessert import SummerDessert

class Icecream(SummerDessert):
    """
    ice-cream in Totoro's summer dessert shop
    """
    
    base_unit_price = 7
    
    def dessert_type(self):
        print 'ice-cream!'
        
        
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