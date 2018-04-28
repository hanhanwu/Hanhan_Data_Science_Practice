'''
Created on Apr 28, 2018

This module purchase the summer desserts
'''
from summer_dessert_children import Icecream, Galeto, FrozenYogurt

def main():
    my_icecream = Icecream(has_waffle=True, has_bubble=True, flavor_count=3, discount_price=0.0)
    my_icecream.dessert_type()
    icecream_price = my_icecream.purchase_price()
    print 'ice-cream price:', icecream_price
    
    my_galeto = Galeto(has_waffle=False, has_bubble=False, flavor_count=2, discount_price=0.7)
    my_galeto.dessert_type()
    galeto_price = my_galeto.purchase_price()
    print 'galeto price:', galeto_price
    
    my_frozenyogurt = FrozenYogurt(has_waffle=True, has_bubble=False, flavor_count=1, discount_price=0.9)
    my_frozenyogurt.dessert_type()
    frozenyogurt_price = my_frozenyogurt.purchase_price()
    print 'frozen yogurt price:', frozenyogurt_price
if __name__ == "__main__":
    main()
