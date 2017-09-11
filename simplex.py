from __future__ import division
import numpy as np
import pandas as pd

def calculate_relative_cost(T, c, base):
	"""
	Calculate the zj for the tableau in the iteration j
	The relative cost is the value of the basic variables on the obj func * their line in the tableau
	
	"""
	
	zj = np.zeros(c.shape[0])
	for j in range(base.shape[0]):
		zj = zj + c[base[j]-1]*T[j]
	return zj

def check_unbounded(pivot_col, zero):
	"""
	the numpy masked function can check a condition for the whole array. So we will check if all the elements in the
	pivot column are less or equal to zero (a value of "zero" (a toletance) will be provided)
	
	>>> ma.masked_where(a <= 2, a)
	  masked_array(data = [-- -- -- 3],
	  mask = [ True  True  True False],
	  fill_value=999999)
	  
	Returns True if unbounded, False if not
	"""
	mask = np.ma.masked_where(pivot_col <= zero, pivot_col)
	
	if (mask.count() == 0):
		return True
	else:
		return False
	
def get_solution_ordered(base, b):
	
	sol = np.zeros(b.shape)
	print(b)
	print(base)
	for i in range(b.shape[0]):
		try:
			sol[i] = (b[base[i] - 1][0])
		except IndexError:
			print('basic variables contains a slack var')
		print(base[i] - 1)
		print(b[base[i] - 1][0])
	return sol

def print_tableau(T, b, theta, c, base, cjzj, iteration):
	from IPython.display import display, HTML
	
	Tpd = pd.DataFrame(T)
	bpd = pd.DataFrame(b)
	thetapd = pd.DataFrame(theta)
	cjzjpd = cjzj.reshape(1,-1)
	cjzjpd = pd.DataFrame(cjzjpd)
	cjzjpd = cjzjpd.rename({0: "cjzj"})
	
	tableau = pd.concat([Tpd, bpd, thetapd], axis=1, ignore_index = True)
	tableau = tableau.append(cjzjpd)

	
	for i in range(base.shape[0]):
		tableau = tableau.rename({ i: str(c[base[i] - 1])+"x_"+str(base[i])})
	for i in range(tableau.shape[1]):
		if i < tableau.shape[1] - 2:
			tableau = tableau.rename(columns = {i : "x_"+str(i + 1)})   
	tableau.columns.values[tableau.shape[1] - 2] = "RHS"
	tableau.columns.values[tableau.shape[1] - 1] = "Theta"
		
	
	
	print("Tableau", iteration + 1)
	display(tableau)
	
	
def simplex_phase2 (A, b, c):
	""" 
	On the form of a min problem:
	Ax<=b
	Since it is the phase 2, it assumes the problem is already in a feasible starting position
	Therefore, no artificial variables will be created
	"""
	
	A = np.asarray(A).astype(float)
	b = np.reshape(b, (-1, 1)).astype(float) #make it a column array
	c = np.asarray(c).astype(float)
	zero = 0.0000001
	
	
	# create the slack variables
	# for each restriction in A (i.e. for each line in A) we will add a slack variable
	# in the end, it will be like concatenating A and the Identity Matrix
	try:
		A_rows, A_cols = A.shape
	except ValueError:
		raise ValueError("A must be 2 dimensional")
	
	# the slack variables will be from the number of columns + 1 adding the number of a rows
	# for example, supose A is a 2x3 matrix
	# the slacks will be x4 and x5
	# x_column+1 to x_column+1+rows
	slacks = np.arange(A_cols + 1, A_cols + 1 + A_rows)
	
	# the initial basic variables are the slacks
	base = np.arange(A_cols + 1, A_cols + 1 + A_rows)
	
	I = np.identity(A_rows)
	
	# create the tableau. Which the first line is the cost function
	# then A concat. I concat b
	
	T = np.concatenate((A, I), axis=1)
	
	# add the cost of the slack variables (0) to the cost array
	
	c = np.concatenate((c, np.zeros(A_rows)), axis=0)
	
	# now, we start the loop!
	# the steps are:
	# calculate cj-zj (the relative cost) to figure out which var is added to the base
	# then, get the smallest step to figure out which var will come out of the base
	
	
	solved = False
	solutionType = None
	i = 0
	while not solved:
		cjzj = c - calculate_relative_cost(T, c, base)
		
		# as we are trying to minimize the cost
		# we must get the var with the most negative value
		# if no non basic var is < 0, then we arrived at end of the simplex method
		
		if cjzj[cjzj.argmin()] >= 0:
			print("Solved\ncjzj = ", cjzj)
			solved = True
			solutionType = 'optimal'
			print(solutionType)
			print_tableau(T, b, theta, c, base, cjzj, i)
			
		else: 
			#print(b)
			#print(A)
			
			#print("cjzj = ", cjzj)
			new_basic = cjzj.argmin() + 1
			pivot_col = T[:, [new_basic - 1]]
			#print(pivot_col.reshape(-1,1))
			
			if (not check_unbounded(pivot_col, zero)):
				
				theta = b/pivot_col.reshape(-1,1)
				theta = np.absolute(theta)
				new_nonbasic = base[theta.argmin()]
				base[theta.argmin()] = new_basic
				
				pivot_denominator = pivot_col[theta.argmin()][0]
				#print(pivot_denominator)
				
				# devide pivot row by the pivot_denominator
				
				pivot_row = T[theta.argmin()]
				pivot_row = pivot_row/pivot_denominator
				pivot_b = b[theta.argmin()]/pivot_denominator
				#print(pivot_b)
				#print(b[theta.argmin()])
				
				# using gauss jordan
				# row = row - row[pivot_col]*pivot_row
				for rows in range(T.shape[0]):
					b[rows] = b[rows] - T[rows][new_basic - 1]*pivot_b
					T[rows] = T[rows] - T[rows][new_basic - 1]*pivot_row
				
				T[theta.argmin()] = pivot_row
				b[theta.argmin()] = pivot_b
				
				
				#print("Pivot row:\n", pivot_row)
				#print("cjzj\n", cjzj)
				#print("theta:\n", theta)
				#print("Base:\n", base)
				#print("Tableau:\n", T)
				#print("rhs.:\n", b)
				print_tableau(T, b, theta, c, base, cjzj, i)
				i = i + 1
				
				if i >= 10:
					solved = True
					print("max int reached")
			
			else:
				solved = True
				solutionType = "unbounded"
				print(solutionType)

def dual_simplex(A, b, c):
	""" 
	On the form of a min problem:
	Ax<=b

	if any value in the vector b is < 0 the problem is in a infeasable position
	therefore, we must solve its dual problem until all elements of b >= 0
	"""


	# First, get the most negative value in order to get the variable that leaves the basis
	pass
