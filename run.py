import numpy as np
import matplotlib.pyplot as plt


N = input();
M = input();

rewards = []




Q = np.ones((N,N))

V = np.zeros((N*N,4)) # to store the vaLue of all possible (state,action) pair

V[0][0] = -1000
V[0][3] = -1000

V[N-1][0]= -1000
V[N-1][1] = -1000

V[N*N-1][0] = 100
V[N*N-1][1] = 100
V[N*N-1][2] = 100
V[N*N-1][3] = 100

V[N*N -N][3]=-1000
V[N*N-N][2]= -1000

R = np.zeros((N,N))


for i in range(0,M):
	a= np.random.randint(2,N*N-2)#the bottom rightmost grid cant have a hole
	r=a/N
	c=a%N
	Q[r][c]=0
	R[r][c]=-100

R[N-1][N-1]=100


alpha = 0.7
lamda = 0.9

lwr_bnd =0.01
uppr_bnd =1

thrshld = 1

def select_Action(s):
	a = 0
	if(np.random.uniform(0,1) <  thrshld ):
		a =  np.random.randint(0,4)
	else:
		a =  np.argmax(V[s,:])
	return a




def next_state(s,a):
	c= np.random.randint(0,3)
	if(a==0):
		s2=s-N
	if(a==1):
		s2=s+1
	if(a==2):
		s2=s+N
	if(a==3):
		s2=s-1

	if ((s%N==0) and (a==3)):		#left side edge
		if(c==0):
			s2=s-N
		if(c==1):
			s2=s+1
		if(c==2):
			s2=s+N

	if(s%N==N-1) and (a==1):    # right side edge
		if(c==0):
			s2=s-N
		if(c==1):
			s2=s-1
		if(c==2):
			s2=s+N

	if(s/N == 0) and (a==0):   # upper edge
		if(c==0):
			s2=s+1
		if(c==1):
			s2=s-1
		if(c==2):
			s2=s+N

	if(s/N == N-1) and (a==2):   # lower edge
		if(c==0):
			s2=s-N
		if(c==1):
			s2=s-1
		if(c==2):
			s2=s+1	

	if(s==0 and ( a==0 or a==3)): # upper left corner
		c=np.random.randint(0,2)
		if(c==0):
			s2=s+N
		else:
			s2=s+1

	if(s==N-1 and (a==0 or a==1)): # upper right corner
		c=np.random.randint(0,2)
		if(c==0):
			s2=s+N
		else:
			s2=s-1

	if(s==N*N-1 and (a==1 or a==2)): # lower right corner
		c=np.random.randint(0,2)
		if(c==0):
			s2=s-N
		else:
			s2=s-1

	if(s== N*N-N and (a==3 or a==2)):  # lower left corner
		c=np.random.randint(0,2)
		if(c==0):
			s2=s+1
		else:
			s2=s-N

	return s2;


def update(s,s1, r, a):
	V[s][a] = (1-alpha)*V[s][a] + alpha*(r + lamda*np.max(V[s1,:] ) )


#BEGIN
max_steps =  200
max_episodes =  8000

for episodes in range(0,max_episodes):

	state  = 0 # reset the state for each episode
	reward = 0

	for steps in range(0,max_steps):

		a=select_Action(state)
		#print(state)
		state2 = next_state(state,a)
		update(state, state2 , R[state2/N][state2%N], a)
		reward =  reward + R[state2/N][state2%N]
		state = state2
		if (state == N*N-1):
			break;

	#inner loop ends
	rewards.append(reward)
	thrshld = lwr_bnd + (uppr_bnd - lwr_bnd)*np.exp(-0.06*episodes)
	# print(" ")
	# print(" ")
	# print(V)
	



print("Final_Path")

thrshld =0
state  = 0 # reset the state for each episode
for steps in range(0,max_steps):
	a=select_Action(state)
	print(state)
	state2 = next_state(state,a)
	update(state, state2 , R[state2/N][state2%N], a)
	state = state2
	if (state >= N*N-1):
		break;


plt.xlabel(" Episodes ")
plt.ylabel(" Rewards ")
epi = range(0,max_episodes) 
plt.plot( epi , rewards, '-b')
plt.show()



