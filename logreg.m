% printing option
more off;

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);
% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];
learning_rate_array = [1e-0,1e-2,1e-4,1e-6];
 
% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
tolerance = 1e-2;                    % tolerance for stopping criteria
max_iter = 1000;                     % maximum iteration
%Iteration for different leaning rates
disp('Non-Regularized model')
for lr = 1: length(learning_rate_array)
    w = zeros(n_vars, 1);                % initialize parameter w
    iter = 0;                            % iteration counter
    int_count = 0;
    learning_rate = learning_rate_array(lr);%learning_rate_array(lr);                % learning rate
    disp('-------------------------------------------------')
    fprintf('Learning rate = %d\n', learning_rate);
    test_accuracy_array = {};
    train_accuracy_array = {};
    while true
        iter = iter + 1;                 % start iteration
        int_count = int_count+1;
        %disp(iter)
        train_exp_vector = X_tr(:,:)*w;
        %calculate gradient
        grad = zeros(n_vars, 1);         % initialize gradient
        for j=1:n_vars
            for i = 1:n_tr
                grad(j) = grad(j) + (y_tr(i)-(1/(1+exp(-train_exp_vector(i)))))*X_tr(i,j);  % compute the gradient with respect to w_j here
            end
        end
        
        %take step
        w_new = w + (learning_rate*grad);            % take a step using the learning rate
        if mod(iter,50) == 0
            fprintf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
        end
        %stopping criteria and perform update if not stopping
        if mean(abs(grad)) < tolerance
            disp('tolerated W')
            w = w_new;
            break;
        else
            w = w_new;
        end
        %disp(w)

       %use w for prediction 
        if int_count == 50
            %Testing Accuracy
            test_exp_array = X_ts(:,:)*w;
            pred_ts = zeros(n_ts, 1);               % initialize prediction vector
            count1 = 0;
            for i=1:n_ts
                if (1/(1+exp(test_exp_array(i)))) < (1/(1+exp(-test_exp_array(i))))
                    pred_ts(i) = 1;            % compute prediction
                else
                    pred_ts(i) = 0;
                end
                if pred_ts(i) == y_ts(i)
                    count1 = count1 + 1;
                end
            end
            test_accuracy_array = [test_accuracy_array,(count1/n_ts)];

            %Training Accuracy
            pred_tr = zeros(n_tr, 1);               % initialize prediction vector
            count2 = 0;
            for i=1:n_tr
                if (1/(1+exp(train_exp_vector(i)))) < (1/(1+exp(-train_exp_vector(i))))
                    pred_tr(i) = 1;            % compute prediction
                else
                    pred_tr(i) = 0;
                end
                if pred_tr(i) == y_tr(i)
                    count2 = count2 + 1;
                end
            end
            train_accuracy_array = [train_accuracy_array,(count2/n_tr)];
            int_count = 0;

        end

        if iter >= max_iter 
            break;
        end

    end
    % Plot Training and Testing Accuracy for different learning rates
    figure('Name',strcat('Training and Testing Accuracy for learning rate = ',num2str(learning_rate_array(lr))),'NumberTitle','off')
    plot([50:50:1000],cell2mat(test_accuracy_array),'b',[50:50:1000],cell2mat(train_accuracy_array),'r')
    xlabel('No of Iterations')
    ylabel('Training/Testing Accuracy')
    legend('Testing Accuracy','Training Accuracy')
    %for part b output
    if learning_rate == 1e-4
        max_tr_acc = train_accuracy_array{length(train_accuracy_array)};
        max_ts_acc = test_accuracy_array{length(test_accuracy_array)};
    end
end

fprintf('\n')
fprintf('\n')

% Using Regularization
disp('Using Regularization')
reg_test_accuracy_array = {};
reg_train_accuracy_array = {};
k_array = [-8,-6,-4,-2,0,2,4,6,8];
for k = 1: length(k_array)
    fprintf('Value of k = %d\n',k_array(k))
    w_reg = zeros(n_vars, 1);                % initialize parameter w
    lambda = 2^(k_array(k));                   %set Lambda
    iter = 0;                            % iteration counter
    int_count = 0;
    learning_rate = 1e-3;              % learning rate
    fprintf('Learning rate = %d\n',learning_rate)
    while true
        iter = iter + 1;                 % start iteration
        int_count = int_count+1;
        train_exp_vector = X_tr(:,:)*w_reg;     %dot product vector
        %calculate gradient
        grad_reg = zeros(n_vars, 1);         % initialize gradient
        for j=1:n_vars
            for i = 1:n_tr
                grad_reg(j) = grad_reg(j) + (y_tr(i)-(1/(1+exp(-train_exp_vector(i)))))*X_tr(i,j);  % compute the gradient with respect to w_j here
            end
            %Regularization
            grad_reg(j) = grad_reg(j)-(lambda*w(j)); 
        end

        %take step
        w_new = w_reg + (learning_rate*grad_reg);            % take a step using the learning rate
        if (mod(iter,50)) == 0
            fprintf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad_reg)));
            fprintf('\n')
        end
        %stopping criteria and perform update if not stopping
        if mean(abs(grad_reg)) < tolerance
            disp('tolerated W')
            w_reg = w_new;
            break;
        else
            w_reg = w_new;
        end
        %disp(w_reg)
        
        if iter >= max_iter 
            break;
        end

    end
    %use w for prediction 
        %Reg Testing Accuracy
        pred_ts = zeros(n_ts, 1);               % initialize prediction vector
        count1 = 0;
        test_exp_array = X_ts(:,:)*w_reg;       % dot product vector
        for i=1:n_ts
            if (1/(1+exp(test_exp_array(i)))) < (1/(1+exp(-test_exp_array(i))))
                pred_ts(i) = 1;            % compute your prediction
            else
                pred_ts(i) = 0;
            end
            if pred_ts(i) == y_ts(i)
                count1 = count1 + 1;
            end
        end
        reg_test_accuracy_array = [reg_test_accuracy_array,(count1/n_ts)];

        %Reg Training Accuracy
        pred_tr = zeros(n_tr, 1);               % initialize prediction vector
        count2 = 0;
        for i=1:n_tr
            if (1/(1+exp(train_exp_vector(i)))) < (1/(1+exp(-train_exp_vector(i))))
                pred_tr(i) = 1;            % compute your prediction
            else
                pred_tr(i) = 0;
            end
            if pred_tr(i) == y_tr(i)
                count2 = count2 + 1;
            end
        end
        reg_train_accuracy_array = [reg_train_accuracy_array,(count2/n_tr)];
        
end
% Plot Training and Testing Accuracy for 0.0001 learning rate and different values of K 
figure('Name',strcat('Training and Testing Accuracy for 0.0001 learning rate and different values of K'),'NumberTitle','off')
plot(k_array,cell2mat(reg_test_accuracy_array),'b',k_array,cell2mat(reg_train_accuracy_array),'r')
xlabel('K Values')
ylabel('Training/Testing Accuracy')
legend('Testing Accuracy','Training Accuracy')
fprintf('\t      Training acc \t Testing_acc\n')
fprintf('Reg \t  %d \t %d \n',reg_train_accuracy_array{8},reg_test_accuracy_array{8})
fprintf('Non-Reg   %d \t %d \n',max_tr_acc,max_ts_acc)

