# evolution.py - Probability density evolution 
# ver 1.2 - 2024-02-15 - michael.muskulus@ntnu.no

# CHANGELOG
#
# 2024-02-15
# - Added search for changes in optimal stock when varying alpha
# - Examples numbered according to proceedings
# - New function optimize_and_compare to avoid repetition
# - Improved textual outputs slightly
# - Increase portability by replacing unpacking with tuples
# - Add plotting as PDFs in addition to PNGs

import numpy as np
import matplotlib.pyplot as plt
import queue
import matplotlib.animation as animation
from scipy.stats import poisson, nbinom
import scipy.stats.contingency as ssc

#
# Models for customer behavior
#

# Return a dictionary with the following elements:
#
#   fi : Customer preference distribution (must add to one)
#        Index j = 1,2,...,m (different purchases)
#   bij: Items demanded (multiset)
#        Index i = 1,2,...,m (different purchases)
#        Index j = 1,2,...,n (different items)
#   aij: Switching distribution 
#        Index i = 1,2,...,m (different purchases)
#        Index j = 1,2,...,m (different purchases)
# Here we use:
#   m:   Number of different purchases
#   n:   Number of different items available

# SimpleCustomer
# - Single item purchase
# - Preference distribution (different customers / wishes)
# FlexibleCustomer
# - Single item purchase
# - Preference distribution
# - Switching probability (one switch possible)
# CorrelatedCustomer
# - Multi-buy purchase (a single multiset)
# - Modeled by multiset
# - No switching possible
# GeneralCustomer
# - Multi-buy purchases (multisets)
# - Modeled by multiset
# - Switching probability (one switch possible)

def customerSimple(mu, f, fi):
    """
    Single item customer with no switching
    n = length of fi
    m = n    
    f : Customer appearance distribution
    fi: Customer item preferences
    """
    fi = np.array(fi)
    nx = len(fi)
    # assert(len(fi) == nx)  
    assert(np.isclose(np.sum(fi),1))
    aij = np.zeros((nx,nx))
    for i in range(nx):
        aij[i,i] = 1 - np.sum(aij[i,:])  # Probability of not switching
    bij = np.zeros((nx,nx),dtype='int')    
    np.fill_diagonal(bij, 1)  # Single item purchases
    return {'mu': mu, 'f': f, 'fi': fi, 'aij': aij, 'bij': bij}

def customerFlexible(mu, f, fi, aij):
    """
    Single item customer with switching
    n = length of fi
    m = n    
    f  : Customer appearance distribution
    fi : Customer item preferences
    aij: Switching matrix
    """
    fi = np.array(fi)
    nx = len(fi)
    # assert(len(fi) == nx)
    assert(np.isclose(np.sum(fi),1))
    bij = np.zeros((nx,nx),dtype='int')    
    np.fill_diagonal(bij, 1)  # Single item purchases
    aij = np.array(aij)
    assert(aij.shape == (nx,nx))
    for i in range(aij.shape[0]):
        assert(np.isclose(np.sum(aij[i,:]), 1.))
    return {'mu': mu, 'f': f, 'fi': fi, 'aij': aij, 'bij': bij}

def customerCorrelated(mu, f, bi):
    fi = np.ones(1)  # only one entry
    bij = np.array(bi)
    nx = len(bij)
    bij = bij.reshape((1,nx))  # present as matrix
    bij = bij.astype('int')
    aij = np.ones((1,1))
    return {'mu': mu, 'f': f, 'fi': fi, 'aij': aij, 'bij': bij}

def customerGeneral(mu, f, fi, aij, bij):
    fi = np.array(fi)
    m = len(fi)
    assert(np.isclose(np.sum(fi),1))
    bij = np.array(bij).astype('int')
    aij = np.array(aij)
    nx = aij.shape[0]
    assert(aij.shape == (nx,nx))
    assert(bij.shape == (m,nx))
    for i in range(aij.shape[0]):
        aij[i,i] = 1 - np.sum(aij[i,:])  # Probability of not switching    
    return {'mu': mu, 'f': f, 'fi': fi, 'aij': aij, 'bij': bij}
    

#
# Models for customer arrival distributions
#

def dPoisson(mu):
    return lambda k: 1 - poisson.cdf(k-1, mu=mu)

def dNBinom(mu, b=0.99):
    N = mu*b/(1-b)
    return lambda k: 1 - nbinom.cdf(k-1, N, b)

#
# Determine demand distribution
#

# Probability density evolution (with large enough stock)
# How large is large enough? 
# Try with a guess, then increase, if necessary


def evolve_demand(qmax,customers,maxcustomers=10000,minp=1e-9,mind=1e-12,verbose=False):
    # Assumes large enough initial stocks q0
    # Fails if this is not the case    
    # minp: Min. probability of customer existing considered
    # mind: Min. probability of certain demand considered
    nx = len(qmax) 
    qi = np.zeros(qmax+1)
    q0 = np.zeros(nx,dtype='int')
    qi[tuple(q0)]  = 1.0
    # Consider all customer types individually
    for customer in customers:
        f = customer["f"]
        fi = customer["fi"]
        bij= customer["bij"]
        # aij= customer["aij"]  # not necessary for determining demand
        k = 0
        while True:
            k = k + 1  # k-th customer     
            pe = f(k)
            if verbose:
                print("Customer #%3d -- exists with p = %12.10f" % (k, pe))
            if pe < minp or k > maxcustomers:
                break
            # Initialize new demand distribution -- qi_ = np.zeros(q0+1)
            # Case 1: customer does not exist, with probability 1-pe
            qi_ = (1-pe)*qi  # demand not affected
            # Case 2: customer does exist, with probability pe            
            it = np.nditer(qi, flags=['multi_index'])
            # Iterate over all non-zero entries of current demand distribution
            for x in it:    # x = qi[it.multi_index]
                if x < mind:  # insignificant chance of those demand levels
                    continue
                q  = np.copy(it.multi_index)
                for i in range(len(fi)):  # possible customer choices
                    p = pe * fi[i] * qi[tuple(q)]
                    if p < mind:
                        continue
                    d = bij[i]  # customer demands these items
                    q1 = q + d  # new total demand 
                    possible = np.all(q1 <= qmax)                    
                    if not possible:  # domain too small
                        return False, qi  # just fail 
                    # push probability into the new demand
                    qi_[tuple(q1)] = qi_[tuple(q1)] + p
                    if verbose:
                        print("  q%s = %12.10f -- push %12.10f => q%s" % (q,qi[tuple(q)],p,q1), end='\n')
            # check that we do not loose / create probability
            qi = qi_
            total = np.concatenate(qi).sum()
            if verbose:
                print("Total probability pushed forward: %12.10f" % total)
            assert(np.isclose(total, 1.0))    
            # now next customer
        # now next customer type
    return True, qi

def find_demand(dmax,customers):
    qmax = np.array(dmax,dtype='int')
    success, dij = evolve_demand(qmax,customers)
    while not success:
        qmax = qmax*2
        success, dij = evolve_demand(qmax,customers)
    return dij
    
def analyze_demand(dij):
    nx = len(dij.shape)
    # marginals
    di = ssc.margins(dij)
    # maximum (conative bound)
    dmax = np.zeros(nx,dtype='int')
    for i in range(nx):
        dmax[i] = np.max(np.where(di[i] > 0))
    # expectation
    # use marginals for this
    avg = np.zeros(nx)
    for i in range(nx):
        dj = di[i].flatten()
        for j in range(len(dj)):
            avg[i] = avg[i] + j*dj[j]
    # maximum probability = most likely
    mpp = np.argmax(dij)
    mpp = np.array(np.unravel_index(mpp, dij.shape))
    return {'di': di, 'dmax': dmax, 'mpp': mpp, 'avg': avg}

#
# Determine actual sales
#

def evolve_correlated(q0,customers,maxcustomers=10000,minp=1e-8,mind=1e-12,verbose=False):
    # Evolve the stock distributions for a series of customers
    # Sales we do need not model them explicity, since we can infer them
    # from the remaining stocks
    # minp: Min. probability of customer existing considered
    # mind: Min. probability of certain demand considered
    def buy(q,items,p,qi,qi_,usc,usc_):
        # Buy item(s) from stock q with probability p
        # Update qi and usc
        q1 = q - items                   # Reduced stock after sale
        assert(np.all(q >= 0))           # Sale should be possible
        qi_[tuple(q1)] = qi_[tuple(q1)] + p          # Probability for new stock level
        if verbose:
            print("      q%s = %12.10f -- push %12.10f => q%s" % (q,qi[tuple(q)],p,q1), end='\n')
        # Update distribution of unsatisfied customers
        usc_[:] = usc_[:] + p * usc      # No change       
    def leave(q,p,qi,qi_,usc,usc_):
        # No sale, with probability p
        qi_[tuple(q)] = qi_[tuple(q)] + p            # Probability for same stock level
        if verbose:
            print("      q%s = %12.10f -- push %12.10f => q%s (lost sale)" % (q,qi[tuple(q)],p,q), end='\n')
        # One more unsatisfied customer
        usc_[1:] = usc_[1:] + p * usc[:-1]
    def loosesales(p,items):
        pmoved = p * notlostsales 
        lostsales[items]    = lostsales[items]    + pmoved[items]
        notlostsales[items] = notlostsales[items] - pmoved[items]
    nx = len(q0)
    # Keep track of stock levels
    qi = np.zeros(q0+1)
    qi[tuple(q0)]  = 1.0  # Initial stock level
    # Keep track of unsatisfied customers
    usc = np.zeros(maxcustomers) 
    usc[0] = 1.0  # All customers potentially satisfied
    # Keep track of lost sales / no lost sale probabilities
    lostsales    = np.zeros(nx)
    notlostsales = np.ones(nx)
    k = 0
    while True:
        k = k + 1  # k-th potential customer     
        # Consider all customer types individually
        # for customer in customers:
        pmin = 1.0  # Smallest pe
        for ic in range(len(customers)):
            customer = customers[ic]
            f = customer["f"]
            fi = customer["fi"]
            bij= customer["bij"]
            aij= customer["aij"]
            # print("aij = ")
            # print(aij)
            pe = f(k)
            pmin = min(pe,pmin)
            if verbose:
                print("Customer #%3d (type %d) -- exists with p = %12.10f" % (k, ic, pe))
            if pe < minp:
                break
            # Case 1: customer does not exist, with probability 1-pe
            qi_ = (1-pe)*qi  
            usc_ = (1-pe)*usc
            # Case 2: customer does exist, with probability pe            
            # Iterate over all non-zero entries of current demand distribution
            it = np.nditer(qi, flags=['multi_index'])
            for x in it:    # x = qi[it.multi_index]
                if x < mind:  # Insignificant chance of those demand levels
                    continue
                q  = np.copy(it.multi_index)
                if verbose:
                    print("  Consider q%s = %12.10f" % (q, qi[tuple(q)]))
                for i in range(len(fi)):  # All possible customer choices
                    p = pe * fi[i] * qi[tuple(q)]
                    if p < mind:
                        continue
                    # First choice
                    items = bij[i]
                    if verbose:
                        print("  First choice ",i," with p =",p," items = ",items)
                    possible = np.all(q >= items)
                    if possible:
                        # Case 2a: make the sale
                        buy(q,items,p,qi,qi_,usc,usc_)
                    else:
                        # Case 2b: consider all possibilities for switching                    
                        nalternatives = len(aij[i])  # no. alternatives
                        for j in range(nalternatives):
                            palt = p * aij[i,j]  
                            if palt < mind:
                                continue
                            items = bij[j]
                            # if j==i the sale is not possible anyways
                            possible = np.all(q >= items)
                            if verbose:
                                print("    Alternative  ",j," with p =",palt, " items = ",items)
                            if possible:
                                # Case 2b1: make the sale
                                buy(q,items,palt,qi,qi_,usc,usc_)
                            else:
                                if verbose and (i == j):
                                    print("    Primary      ",j," with p =",palt, " items = ",items)
                                # Case 2b2: lost sale
                                leave(q,palt,qi,qi_,usc,usc_)
                                loosesales(palt,items)
            qi = qi_
            usc = usc_
            # Check that we do not loose / create probability
            total = np.concatenate(qi).sum()
            total_usc = np.sum(usc)
            if verbose:
                print("Total probability stock levels         : %12.10f" % total)
                print("Total probability unsatisfied customers: %12.10f" % total_usc)
            assert(np.isclose(total, 1.0))    
            assert(np.isclose(total_usc, 1.0))    
            # now next customer
            # assert(1==2)
        if pmin < minp or k >= maxcustomers:
            break  # pe too small across all customer types
        # now next customer type
    if verbose:
        print("Total no. customers simulated: %d" % k)
    return qi, usc, lostsales

#
# Analyze results
#

def analyze(qi, usc, lostsales, q0, costs, prices, minp=1e-12, verbose=True):
    # Analyze the eventual stock distribution
    # - How many sales?
    # - How much revenue?
    esv  = 0.0       # expected sale value, add to it
    esv2 = 0.0       # square of esv, for variance calculation
    esales  = 0.0    # expected no. sales
    esales2 = 0.0    # square of esales, for variance calculation
    eusc  = 0.0      # expected no. unsatisfied customers
    eusc2 = 0.0      # square of eusc, for variance calculation
    ptotal = 0.0     # total probability of stock levels
    pcost = np.dot(q0,costs)  # production costs (scalar product)
    it = np.nditer(qi, flags=['multi_index'])
    for x in it:     # x = qi[it.multi_index]
        if x < minp:
            continue
        # x: probability of having final stock q
        q = np.copy(it.multi_index)
        sales = q0 - q
        nsales = np.sum(sales)
        value  = np.dot(sales,prices)
        esv  = esv + value * x  
        esv2 = esv2 + value**2 * x
        esales  = esales + nsales * x
        esales2 = esales2 + nsales**2 * x
        ptotal = ptotal + x
    if verbose:
        print("Initial stock level           : q0 =",q0)
        print("Total probability analyzed    : %12.10f" % ptotal)
    assert(np.isclose(ptotal,1.0))            
    maxcustomers = len(usc)
    for i in range(maxcustomers):
        eusc  = eusc  + i * usc[i]
        eusc2 = eusc2 + i**2 * usc[i]
    if verbose:
        esvstd    = np.sqrt(esv2 - esv**2)        # E[X^2] - E[X]^2
        esalesstd = np.sqrt(esales2 - esales**2) 
        euscstd   = np.sqrt(eusc2 - eusc**2) 
        print("Average sales                 : %3d +/- %4.2f" % (esales, esalesstd))
        print("Average unsatisfied customers : %3d +/- %4.2f" % (eusc, euscstd))
        print("Lost sales prob. (per item)   :", lostsales) 
        print("Average sales value           : %8.4f +/- %8.4f" % (esv, esvstd))
        print("Production cost               : %8.4f" % pcost)
        print("Expected revenue              : %8.4f +/- %8.4f" % (esv - pcost, esvstd))
    return esv-pcost  # expected revenue

#
# Theoretical solution for independent demand
#

def theoretical_q(customers, costs, prices, minp=1e-9, verbose=False):
    nx = len(costs)
    f = (prices-costs)/prices
    mu_eff = np.zeros(nx)  # effect demand intensity for each item
    if verbose:
        print("Theoretical solution")
    for customer in customers:      
        if verbose:
            print("Analyze customer")
        mu = customer["mu"]
        fi = customer["fi"]
        bij = customer["bij"]
        if verbose:
            print("  fi = ",fi)
        for i in range(len(fi)):
            if fi[i] < minp:
                continue
            items = bij[i]
            if verbose:
                print("  Choice ",i, " has items = ", items)
            for k in range(len(items)):
                if items[k] > 0:
                    # demand for item k with intensity mu*fi[i]
                    if verbose:
                        print("    Increase demand for item ",k," with intensity",mu*fi[i])
                    mu_eff[k] = mu_eff[k] + mu*fi[i]
    q = poisson.ppf(f,mu_eff)
    # b = 0.99
    # N = mu_eff*b/(1-b)
    # q = nbinom.ppf(f,N,b)
    q[q < 0] = 0
    q = np.round(q)  # NB: is rounding the best strategy?
    q = q.astype(int)
    if verbose:
        print("Effective intensities of items")
        print(mu_eff)
        print("Optimal stock levels")
        print(q)
    return q

#
# Optimization
#

# Step-up stocks until no more improvement
# Keep a queue of unexplored possibilities
# Stop when no improvement can be reached anymore
def optimize_full(q0,qmax,customers,costs,prices,depth=2,quiet=False,verbose=False):
    if not quiet:
        print("Fully enumerative optimization",end="")
    NOT_VISITED = -99999
    def obj(q,penalize_lost=False,penalize_item=1,level=0.01,penalty=10000):        
        qi, usc, lostsales = evolve_correlated(q,customers)
        esv = analyze(qi,usc,lostsales,q,costs,prices,verbose=verbose)
        res = esv
        if penalize_lost:
            if lostsales[1] > level:
                res = res - penalty*(lostsales[penalize_item]-level) 
        return res
    def add(q,current):
        # check if we have been here before
        if vals[tuple(q)] <= NOT_VISITED:
            new = obj(q)
            vals[tuple(q)] = new
            # is there a local improvement?
            if (new > current):
                # new base for further exploration
                if verbose:
                    print("%s -- %12.10f -- PUSHED" % (q, new))
                candidates.put(np.copy(q))
                if not quiet:
                    print(".",end="")
            else:
                if verbose:
                    print("%s -- %12.10f -- <" % (q, new))   
    def add_from(q,depth):
        # try to find new candidates from existing element q
        current = vals[tuple(q)]
        for coord in range(len(q)):
            q[coord] = q[coord] + 1
            add(q,current)  # single step-up
            if depth > 1:
                add_from(q,depth-1)
            q[coord] = q[coord] - 1            
    best = obj(q0)    
    qbest = np.copy(q0)
    candidates = queue.Queue()
    step = 0
    vals = np.ones(qmax)*NOT_VISITED # keeping track of values
    vals[tuple(q0)] = best
    # enter new candidates into queue
    add_from(q0,depth)    
    if not quiet:
        print("\n%s -- %12.10f -- initial solution" % (qbest, best), end="")
    while not candidates.empty():
        q = candidates.get()
        step = step + 1
        # should already have a value
        current = vals[tuple(q)]
        assert(current > NOT_VISITED)
        # but is there a global improvement?
        if current > best:
            best = current
            qbest = np.copy(q)
            if not quiet:
                print("\n%s -- %12.10f -- new best solution" % (qbest, best),end="")
        # now try to find new candidates to explore from this one
        add_from(q,depth)
    if not quiet:
        print("\nFinished... no. evaluations = %d" % step)
    return(qbest)

#
# Some example scenarios
#

def optimize_and_compare(customers,costs,prices):
    print("")
    print("*** Theoretical solution for independent stocks ***")
    qt = theoretical_q(customers,costs,prices)
    qi, usc, lostsales = evolve_correlated(qt,customers)
    analyze(qi,usc,lostsales,qt,costs,prices)
    # Let's try to optimize
    q0 = np.array([0,0])        # initial stock levels to consider
    qmax = np.array([100,100])  # max. stock levels to consider
    print("")
    qbest = optimize_full(q0,qmax,customers,costs,prices)
    qi, usc,lostsales = evolve_correlated(qbest,customers)
    print("")
    print("*** Optimal solution from search ***")
    analyze(qi,usc,lostsales,qbest,costs,prices)
    return qi    

def example4(plot=True,save=False):
    # Single item customer / independent case
    print("Example 4")
    print("Results for simple customer")
    print("")
    print("Demand distribution for simple customer")
    customers = []
    rate = 20
    fi = np.array([2/3, 1/3])
    # fi = np.array([1/2, 1/2])
    
    #customers.append(customerSimple(rate,dPoisson(mu=rate),fi))
    customers.append(customerSimple(rate,dNBinom(mu=rate),fi))
    dmax  = [29,29]  # initial guess of customer demand
    dij = find_demand(dmax,customers) 
    foo = analyze_demand(dij)
    print("  Maximum demand estimated:",foo["dmax"])
    print("  Most probable demand    :",foo["mpp"])
    print("  Mean demand             :",foo["avg"])
    print("")

    if plot:
        print("Plotting the final stock distribution")
        q0 = foo["dmax"]
        qi = np.zeros(q0+1)
        qi[tuple(q0)] = 1.
  
        plt.figure()
        plt.matshow(qi,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.title("Initial stock distribution")
        if save:
            plt.savefig('fig1-0.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig1-0.pdf', bbox_inches='tight', dpi=300)
    
        qi,usc,lostsales = evolve_correlated(q0,customers)
        plt.figure()
        plt.matshow(qi,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.title("Final stock distribution")
        if save:
            plt.savefig('fig1-1.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig1-1.pdf', bbox_inches='tight', dpi=300)

def example5(plot=True,save=False):
    # Single item customer / independent case
    print("Example 5")
    print("Results for simple customer")
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    rate = 20
    fi = np.array([2/3, 1/3])
    customers.append(customerSimple(rate,dNBinom(mu=rate),fi))
    qi = optimize_and_compare(customers,costs,prices)
    return qi

def example5_plot(plot=True,save=False):
    # Single item customer / independent case
    print("Example 5")
    print("Results for simple customer")
    print("")
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    rate = 20
    fi = np.array([2/3, 1/3])
    customers.append(customerSimple(rate,dNBinom(mu=rate),fi))
    # customers.append(customerSimple(rate,dPoisson(mu=rate),fi))

    q0 = np.array([10,5])  # initial stock levels
    if plot:
        print("Plotting the case with q0 =" % q0)
        qi,usc,lostsales = evolve_correlated(q0,customers)  
        analyze(qi, usc, lostsales, q0, costs, prices)

        qi1 = np.zeros((14,6))  # embed in larger array
        qi1[0:11,0:6] = qi 
        plt.figure()
        plt.matshow(qi1,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.clim(0,0.2)
        plt.title("q0=%s" % q0)
        if save:
            plt.savefig('fig2-1.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig2-1.pdf', bbox_inches='tight', dpi=300)
        print("")

    # Let's try to optimize
    q0 = np.array([0,0])        # initial stock levels to consider
    qmax = np.array([100,100])  # max. stock levels to consider
    qbest = optimize_full(q0,qmax,customers,costs,prices)
    qi, usc,lostsales = evolve_correlated(qbest,customers)
    print("")
    print("*** Optimal solution from search ***")
    analyze(qi,usc,lostsales,qbest,costs,prices)
    if plot:
        print("\nPlotting the optimal case q* =" % qbest)
        plt.figure()
        plt.matshow(qi,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.clim(0,0.2)
        plt.title("q0=%s" % qbest)
        if save:
            plt.savefig('fig2-2.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig2-2.pdf', bbox_inches='tight', dpi=300)

    print("")
    print("*** Theoretical solution for independent stocks ***")
    qt = theoretical_q(customers,costs,prices)
    qi, usc, lostsales = evolve_correlated(qt,customers)
    analyze(qi,usc,lostsales,qt,costs,prices)
    if plot:
        print("\nPlotting the independent case q0 ="% qt)
        plt.figure()
        plt.matshow(qi,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.clim(0,0.2)
        plt.title("q0=%s" % qt)
        if save:
            plt.savefig('fig2-3.png', bbox_inches='tight', dpi=300)
            plt.savefig('fig2-3.pdf', bbox_inches='tight', dpi=300)

    return qi

def example6(alpha=1.0):
    # Single item customer with switching probability alpha
    print("Example 6")
    print("Results for simple customer with switching (strict)")
    print("alpha = %6.4f" % alpha)
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    rate = 20
    fi = np.array([2/3, 1/3])
    aij = np.zeros((nx,nx))
    assert(alpha <= 1.0 and alpha >= 0.0)
    aij[0,1] = alpha
    aij[1,0] = alpha
    for i in range(nx):
        aij[i,i] = 1 - np.sum(aij[i,:])  # probability of not switching
    customers.append(customerFlexible(rate,dNBinom(mu=rate),fi,aij))
    qi = optimize_and_compare(customers,costs,prices)
    return qi

def example6_plot(save=False):
    # Single item customer with switching probability
    def aijmatrix(r):
        aij = np.zeros((nx,nx))
        aij[0,1] = r 
        aij[1,0] = r
        for i in range(nx):
            aij[i,i] = 1 - np.sum(aij[i,:])  # probability of not switching
        return aij
    print("Example 6_plot")
    print("Results for simple customer with switching")
    print("Varying the switching probability")
    print("")
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    rate = 20
    fi = np.array([2/3, 1/3])
    
    # rs = np.linspace(0,1,11)
    rs = np.linspace(0,1,101)
    value = np.zeros_like(rs)
    qs = []
    for i, r in enumerate(rs):
        assert(i == len(qs))
        print(i,": Results for r = %12.10f" % r,end="")
        customers = []
        customers.append(customerFlexible(rate,dNBinom(mu=rate),fi,aijmatrix(r)))
        # Let's try to optimize
        q0 = np.array([0,0])        # initial stock levels to consider
        qmax = np.array([100,100])  # max. stock levels to consider   
        qbest = optimize_full(q0,qmax,customers,costs,prices,quiet=True)
        qi, usc,lostsales = evolve_correlated(qbest,customers)
        esv = analyze(qi, usc, lostsales, qbest,costs,prices,verbose=False)
        value[i] = esv
        qs.append(qbest)
        print("  =>  q* =",qbest," value = %6.3f" % esv)
    
    plt.figure()
    plt.title("Optimal revenue")
    plt.plot(rs,value,'-b')
    plt.plot(rs,value,'ko')
    plt.xlabel("Switching probability")
    plt.ylabel("Expectation")
    if save:
        plt.savefig('fig4x-1.png', bbox_inches='tight', dpi=300)
        plt.savefig('fig4x-1.pdf', bbox_inches='tight', dpi=300)
    
    plt.figure()
    plt.title("Optimal stock levels")
    for q in qs:
        plt.plot(q[0],q[1],'ko')
    plt.xlabel("# Item 1")
    plt.ylabel("# Item 2")
    if save:
        plt.savefig('fig4x-2.png', bbox_inches='tight', dpi=300)
        plt.savefig('fig4x-2.pdf', bbox_inches='tight', dpi=300)

    return rs, qs, value

def example6_search(dr=0.1,rtol=1e-8,r1=0.0,r2=1.0,save=False):
    # Single item customer with switching probability
    # Find the values of alpha where the optimum changes
    # dr: Minimum increase in r when searching for next change
    # r1, r2: Search region between r1 and r2
    def aijmatrix(r):
        aij = np.zeros((nx,nx))
        aij[0,1] = r 
        aij[1,0] = r
        for i in range(nx):
            aij[i,i] = 1 - np.sum(aij[i,:])  # probability of not switching
        return aij
    def evaluate(r,quiet=False):
        if not quiet:
            print("  r = %12.10f" % r,end="")
        customers = []
        customers.append(customerFlexible(rate,dNBinom(mu=rate),fi,aijmatrix(r)))
        q0 = np.array([0,0])        # initial stock levels to consider
        qmax = np.array([100,100])  # max. stock levels to consider   
        qbest = optimize_full(q0,qmax,customers,costs,prices,quiet=True)
        qi, usc,lostsales = evolve_correlated(qbest,customers)
        esv = analyze(qi, usc, lostsales, qbest,costs,prices,verbose=False)
        if not quiet:
            print("  =>  q =",qbest)
        return esv, qbest
    def bisect(r1,r2,qlast):
        # Find r between r1 and r2 where q changes
        print("  Bisecting...")
        r = (r1+r2)/2
        while r2-r1 > rtol:
            r = (r1+r2)/2
            v, q = evaluate(r,quiet=True)
            print("    r1= %12.10f -- r= %12.10f -- r2= %12.10f" % (r1, r, r2), "  =>  q =",q)
            if np.all(q == qlast):
                r1 = r 
                continue
            r2 = r 
        return r
    print("Example 6_search")
    print("Results for simple customer with switching")
    print("Varying the switching probability")
    print("")
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    rate = 20
    fi = np.array([2/3, 1/3])
    r = r1
    rs = [r]
    v, q = evaluate(r)
    value = [v]
    qs = [q]
    qlast = np.copy(q)
    rlast = r
    while r < r2:
        r = r + dr
        if r > r2:
            r = r2
        v, q = evaluate(r)
        if np.all(q == qlast):
            rlast = r 
            if np.isclose(r, r2):
                break
            continue  # need to search farther away
        r = bisect(rlast,r,qlast)  # find the value of r where there is a change
        # store results
        v, q = evaluate(r)
        if np.all(q == qlast):
            # no change detected e.g. accuracy too low
            rlast = r
            continue
        print("Change at r = %12.10f" % r,"  =>  q* =",q," value = %6.3f" % v)
        rs.append(r)
        value.append(v)
        qs.append(q)
        rlast = r
        qlast = np.copy(q)
    
    plt.figure()
    plt.title("Optimal revenue")
    plt.plot(rs,value,'-b')
    plt.plot(rs,value,'ko')
    plt.xlabel("Switching probability")
    plt.ylabel("Expectation")
    if save:
        plt.savefig('fig6x-1.png', bbox_inches='tight', dpi=300)
        plt.savefig('fig6x-1.pdf', bbox_inches='tight', dpi=300)
    
    plt.figure()
    plt.title("Optimal stock levels")
    for i in range(len(rs)):
        q = qs[i]
        r = rs[i]
        plt.plot(q[0],q[1],'ko')
        s = "%4.3f" % r
        plt.annotate(s,(q[0]+0.2,q[1]))
    plt.xlabel("# Item 1")
    plt.ylabel("# Item 2")
    if save:
        plt.savefig('fig6x-2.png', bbox_inches='tight', dpi=300)
        plt.savefig('fig6x-2.pdf', bbox_inches='tight', dpi=300)

    return rs, qs, value

def example7(depth=2):
    # Correlated customer
    print("Example 7")
    print("Results for correlated customer (must buy both items)")
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    rate = 20/2
    bi = np.zeros(nx,dtype='int')  # NB: must be integers!
    bi[0] = 1
    bi[1] = 1 
    customers.append(customerCorrelated(rate,dNBinom(mu=rate),bi))
    qi = optimize_and_compare(customers,costs,prices)
    return qi

def example8():
    # General customer
    print("Example 8")
    print("Results for general customer (either item 1 or both items)")
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    rate = 20*2/3
    fi = np.array([0.5,0.5])
    m = len(fi)
    bij = np.zeros((nx,nx)) 
    bij[0,0] = 1
    bij[1,1] = 1 
    bij[1,0] = 1 
    aij = np.zeros([m,m])
    customers.append(customerGeneral(rate,dNBinom(mu=rate),fi,aij,bij))
    qi = optimize_and_compare(customers,costs,prices)
    return qi

def example9():
    # Multiple types of customers
    print("Example 9")
    print("Results for two different types of customers")
    nx = 2
    costs  = np.array([6,10])
    prices = np.array([10,13])
    customers = []
    
    # general customer
    rate1 = 0.5*20*2/3
    fi = np.array([0.5,0.5])
    bij = np.zeros((nx,nx)) 
    bij[0,0] = 1
    bij[1,1] = 1 
    bij[1,0] = 1 
    aij = np.zeros([2,2])
    customers.append(customerGeneral(rate1,dNBinom(mu=rate1),fi,aij,bij))
    # add a correlated customer
    rate = 0.5*20/2
    bi = np.zeros(nx,dtype='int')  # NB: must be integers!
    bi[0] = 1
    bi[1] = 1 
    customers.append(customerCorrelated(rate,dNBinom(mu=rate),bi))
          
    qi = optimize_and_compare(customers,costs,prices)
    return qi

def animate(save=False):
    print("Preparing plots for an animation")
    customers = []
    rate = 20
    fi = np.array([2/3, 1/3])
    #customers.append(customerSimple(rate,dPoisson(mu=rate),fi))
    customers.append(customerSimple(rate,dNBinom(mu=rate),fi))

    q0 = np.array([20,20])
    qi = np.zeros(q0+1)
    qi[tuple(q0)] = 1.
    plt.figure()
    plt.matshow(qi,origin="lower")
    plt.set_cmap('jet')
    plt.colorbar()
    plt.title("Initial stock")
    fname = 'figs-{}.png'.format(0)
    if save:
        plt.savefig(fname, bbox_inches='tight')
    print("Plot no. %d -- %s" % (0, fname))    
    n_frames = 50
    data = np.empty(n_frames, dtype=object)
    data[0] = qi
    import os
    os.chdir("C:\\Users\\muskulus\\Dropbox\\DropSync\\swi2024\\figures")
    for k in range(1,n_frames):
        qi,usc,lostsales = evolve_correlated(q0,customers,maxcustomers=k)
        plt.figure()
        plt.matshow(qi,origin="lower")
        plt.set_cmap('jet')
        plt.colorbar()
        plt.title("Stock after {:2} customer(s)".format(k))
        fname = 'figs-{}.png'.format(k)
        if save:
            plt.savefig(fname, bbox_inches='tight')
        data[k] = qi
        print("Plot no. %d -- %s" % (k, fname))    
    
#
# Main program
#

# Choose an example to run 

# animate()
#qi = example4(plot=True,save=True)
#qi = example5()
qi = example5_plot(save=False)
#qi = example6(alpha=1.0)  
#qi = example6(alpha=0.5)  
#rs, qs, value = example6_plot()
#rs, qs, value = example6_search(dr=0.1,rtol=1e-8)
#rs, qs, value = example6_search(dr=0.1,rtol=1e-8,save=True)
#qi = example7()
#qi = example8()
#qi = example9()



