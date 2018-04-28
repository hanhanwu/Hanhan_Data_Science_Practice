'''
Created on Apr 28, 2018

This module contains an abstract class summer_dessert,
which can be inherented by summer desserts such as ice-cream, galeto, frozen_yogurt, etc.

-- the @abstractmethod here is left empty in the abstract class, so that its child class can do the specific implementation
   You need python built-in class abc to get abstractethod (although the class name is, silly)
'''
from abc import ABCMeta, abstractmethod

class SummerDessert(object):
    """
    Totoro's Summer Desserts Shop
    
    Attributes:
    has_waffle: if it has waffle, gets Ture, alse False
    has_bubble: if it has bubble, gets Ture, alse False
    flavor_count: the number of flavors
    discount_price: the amount of money deducted from the original price
    """
    
    __metaclass__ = ABCMeta
    
    # static variables - can be used by the methods in the class
    base_unit_price = 1
    waffle_price = 0.7
    bubble_price = 0.9
    
    def __init__ (self, has_waffle, has_bubble, flavor_count, discount_price):
        self.has_waffle = has_waffle
        self.has_bubble = has_bubble
        self.flavor_count = flavor_count
        self.discount_price = discount_price
        
        
    @abstractmethod
    def dessert_type(self):
        """
        print out the dessert type
        """
        pass
    
    
    def purchase_price(self):
        """
        The final price that the customer will pay
        """
        purchase_price = self.base_unit_price * self.flavor_count - self.discount_price
        if self.has_waffle == True:
            purchase_price += self.waffle_price
        if self.has_bubble == True:
            purchase_price += self.bubble_price
            
        return purchase_price
