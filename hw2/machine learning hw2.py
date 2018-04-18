
import numpy as np
import matplotlib.pyplot as plt
import math
#from dataNormalization import rescaleNormalization

# Starting codes for the HA2 of CS596

# Fill in the codes between "%PLACEHOLDER#start" and "PLACEHOLDER#end"

# Ground-truth Cashier 
groundUnitPrice = np.array([20, 25, 8]) # for fish, chip, and ketchup, respectively

# step 1: initialize your guess on the unit prices of fish, chip and ketchup.
estimatedUnitPrice = np.array(([10,10,10])) # initial unit prices.
MAX_POSSIBLE_UNIT_PRICE = 50
estimatedUnitPrice = np.random.randint(MAX_POSSIBLE_UNIT_PRICE, size=len(groundUnitPrice))

# choose random initial guesses

#PLACEHOLDER_1#start: set your own stopping conditions and learning rate

#condition 1: maximal iterations, stop.
MAX_ITERATION = 10000
#condition 2: if the difference between your prediction and the cashier's price is smaller than a threshold, stop. 
MIN_DELTA = 0.001
# learning rate
ALPHA = .01#1e-3
#PLACEHOLDER_1#end
MAX_PLACE_ORDER = 10

# Y coordinates for plotting
deltaHistory = []

# step 2: iterative method
for i in range(0, MAX_ITERATION):
    yes_zero = False
    # order a meal (simulating training data)
    
    randomMealPortions = np.random.randint(MAX_PLACE_ORDER, size=3)

    # calculate the estimated price     
    expectedTotalPrice = np.sum(estimatedUnitPrice * randomMealPortions )

    # calculate cashier/true price;     
    cashierPrice = np.sum(groundUnitPrice * randomMealPortions)

    #%%%PLACEHOLDER_2#start
    print('estimated unit price ',estimatedUnitPrice)
    print('ground unit price ', groundUnitPrice)
#    print('random meal portions',randomMealPortions)
#    print('expected total price ',expectedTotalPrice)
#    print('cashier price ',cashierPrice)
    
    # calculate current error
    iterError = expectedTotalPrice - cashierPrice

    # append iterError to the history array
    deltaHistory.append(abs(iterError))
#    print('itererror: ',iterError)
    #update unit prices 
    
#    print('eup[i]: ',estimatedUnitPrice[0])
    
   # print('random ',np.sum(randomMealPortions))
    for g in range(len(estimatedUnitPrice)):
        if g == 0 :#checks for randomMealPortions to see if it is all zero
            if randomMealPortions[g] ==0:
                if randomMealPortions[g+1] == 0:
                    if randomMealPortions[g+2]==0:
                        yes_zero=True;
                        break;
        #change = ALPHA * (iterError/len(estimatedUnitPrice)) *(randomMealPortions[g]/np.sum(randomMealPortions))
        change =ALPHA * iterError *(randomMealPortions[g]/np.sum(randomMealPortions))
        if change < 0:
            change=math.floor(change)
        else:
            change=math.ceil(change)
        estimatedUnitPrice[g] = estimatedUnitPrice[g] -change
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*(estimatedUnitPrice[g]-(cashierPrice*(randomMealPortions[g]/np.sum(randomMealPortions))))
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - (randomMealPortions[g]/np.sum(randomMealPortions))*(expectedTotalPrice-cashierPrice)
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*iterError
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA* (estimatedUnitPrice[g]-groundUnitPrice[g])
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*(1/(2*np.sum(randomMealPortions)))*(iterError)
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*(iterError)*(randomMealPortions[g]/np.sum(randomMealPortions))
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*(estimatedUnitPrice[g]-groundUnitPrice[g])*randomMealPortions[g]
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA * (expectedTotalPrice-cashierPrice)
        #d_of_f = ALPHA *(estimatedUnitPrice[g]-groundUnitPrice[g])*randomMealPortions[g]
        #estimatedUnitPrice[g] = estimatedUnitPrice[g] - ALPHA*(randomMealPortions[g]/np.sum(randomMealPortions))*iterError
    if yes_zero == True:
        continue
    print('estimatedUnitPrice updated: ',estimatedUnitPrice)
    
    #%%%%PLACEHOLDER_2#end
    
    #delta = iterError
    delta=abs(iterError)+np.mean(abs(estimatedUnitPrice - groundUnitPrice))
    #print(delta)
    #check stop conditions
    if abs(delta) < MIN_DELTA:
        break

    print('iteration:{}, delta:{}'.format(i, abs(delta)))


# step 3: evaluation
error = np.mean(abs(estimatedUnitPrice - groundUnitPrice))
print('estimation error:{}'.format(error))

# visualize convergence curve: error v.s. iterations

plt.plot(range(0, len(deltaHistory)), deltaHistory)
plt.xlabel('iteration (cnt:{})'.format(len(deltaHistory)))
plt.ylabel('abs(delta)')
plt.title('Final:{}  est err:{}  actl Δ:{}'.format([ '%.4f' % elem for elem in estimatedUnitPrice ], round(error, 4), round(delta, 4)))
plt.show()


