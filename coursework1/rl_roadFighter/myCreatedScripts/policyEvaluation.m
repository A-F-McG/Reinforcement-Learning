function [ V ] = policyEvaluation( MDP_1, discountFactor_gamma , pi_test1)
%POLICYEVALUATION Returns a matrix V which gives the value of each state

    %Set up an empty matrix of state values
    V = zeros(MDP_1.GridSize);

    %Set up a second empty matrix which will be used to calculate the new value
    %of a specific state using V, the previous values of the states around it 
    temporaryV = zeros(MDP_1.GridSize);

    %Set delta (difference between old value matrix and new value matrix)
    %to anything above theta so that the while loop starts
    delta = 1;

    %set theta to a small positive number (when delta is below this threshold, 
    %decide that policy evaluation has converged enough)
    theta = 0.01;
    
    %initalise counter for the number of iterations this algorithm takes 
    %to converge
    numberOfEvaluationIterations = 0;

    while (delta>theta)
        
    delta = 0;
    
    %for each state
    for i=1:MDP_1.GridSize(1)
        for j=1:MDP_1.GridSize(2)

        %get the possible next states from the current state and
        %the probabilites of going there given the action
        [possibleStatesToTransitionTo, probabilityOfTransitioningToState] = MDP_1.getTransitions([i,j],pi_test1(i,j));
       
        valueOfState = 0;
        
        %calculate the value of the state with action determined by policy by:
        % - calcuating the probability of moving to a next state 
        % - multiplied by the reward of going there 
        % - plus all expected future rewards going there
        % - sum the values of above over each possible future state 
        for k=1:length(probabilityOfTransitioningToState)

             valueOfState = valueOfState+probabilityOfTransitioningToState(k)*((MDP_1.getReward([i,j],possibleStatesToTransitionTo(k,:),1))+discountFactor_gamma*V(possibleStatesToTransitionTo(k,1),possibleStatesToTransitionTo(k,2)));
        end
 
        %update the value of the state in calculatingV, a temporary matrix
        %so that we can still use the old state values for calculations in
        %this loop
        temporaryV(i,j) = valueOfState;
        end

    end
        %calculate delta, the difference between the temporary value matrix and
        %the old one
        delta = max(max(abs(temporaryV - V)));
        
        %set V to the values in the temporary value matrix
        V=temporaryV;
        
        numberOfEvaluationIterations = numberOfEvaluationIterations + 1;
 
    end
    
    %display the number of iterations it took for value matrix to converge
    numberOfEvaluationIterations

end



