%POLICY ITERATION

exercise1;

%initially set policy to false since we have not compared it to anything so 
%don't know if it's stable
policyStable = false;

%counts the number of iterations during policy iteration 
numberOfPolicyIterations = 0;

while (policyStable == false)

    %return the policy evaluation matrix V which gives the value of each state
    V = policyEvaluation( MDP_1, discountFactor_gamma , pi_test1);

    %set the current policy to be called the previous policy 
    previousPolicy=pi_test1;

    %determine an improved policy by doing policy improvement
    pi_test1 = policyImprovement( MDP_1, discountFactor_gamma , pi_test1, V);

    %set the improved policy to be called the new policy 
    newPolicy = pi_test1;   
   
         %if the policy has not changed, then it is an optimal policy so stop 
         %policy iteration
         if (previousPolicy==newPolicy)
             policyStable=true;
         end

    numberOfPolicyIterations = numberOfPolicyIterations + 1;

end

%Display map of car movement trajectory
showCarPlaying;

%Display the number of times the policy iteration algorithm ran
numberOfPolicyIterations

%Display the optimal policy found
pi_test1