function [ pi_test1 ] = policyImprovement( MDP_1, discountFactor_gamma , pi_test1, V)
%POLICYIMPROVEMENT Returns an improved policy pi_test1

    %for every state
    for i=1:MDP_1.GridSize(1)
        for j=1:MDP_1.GridSize(2)

            %initially assume the best action to take is the one specified by
            %the policy so set the state value given the best action to be the
            %current value of the state
            bestActionValue = V(i,j);

            %for each action (UP LEFT, UP, UP RIGHT)
            for action = 1:3

                %get the possible next states from the current state and
                %the probabilites of going there given the action
                [possibleStatesToTransitionTo, probabilityOfTransitioningToState] = MDP_1.getTransitions([i,j],action);

                valueOfState = 0;

                     %calculate the value of the state with specified action by:
                     % - calcuating the probability of moving to a next state 
                     % - multiplied by the reward of going there 
                     % - plus all expected future rewards going there
                     % - sum the values of above over each possible future state 
                     for k=1:length(probabilityOfTransitioningToState)

                         valueOfState = valueOfState+probabilityOfTransitioningToState(k)*((MDP_1.getReward([i,j],possibleStatesToTransitionTo(k,:),action))+discountFactor_gamma*V(possibleStatesToTransitionTo(k,1),possibleStatesToTransitionTo(k,2)));
                     end

                     %if the value of being in a state improves given a
                     %different action to the policy, then change the policy to
                     %that action at that state, and update the value of the
                     %state accordingly
                     if (valueOfState>bestActionValue)
                        pi_test1(i,j) = action;
                        bestActionValue = valueOfState;      
                     end               
            end  

        end
    end
end

