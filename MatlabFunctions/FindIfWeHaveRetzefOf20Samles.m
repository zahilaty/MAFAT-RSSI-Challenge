function [bool_val] = FindIfWeHaveRetzefOf20Samles(sig,num_of_people)
%quick and very ugly
counter = 0;
bool_val = false;
    for ind = 1:1:360
        if sig(ind) == num_of_people
            counter = counter + 1;
        else
            counter = 0;
        end
        if counter == 20
            bool_val = true;
            break
        end
    end
    
end





