with_K = 'tail';
MAE = 0.29;
net = 'DenseNet201';
pred_file = strcat(net, '_', with_K, '_K_MAE_', num2str(MAE), '.txt');
labels_preds_file_path = strcat('../preds/preds_', with_K, '_K/');
labels_preds_file_path = strcat(labels_preds_file_path, pred_file);
s = split(pred_file, '_');
%% labels and preds
labels_and_preds = table2array(readtable(labels_preds_file_path));
labels = labels_and_preds(:, 2);
preds = labels_and_preds(:, 3);
plot(labels);
hold on;
plot(preds);
title('Labels and Preds')
legend('labels', 'preds');
saveas(gcf, strcat('./', 'plots_', with_K, '_K/', net, '_', with_K, '_K_', 'MAE_', num2str(MAE), '_labels_and_preds.jpg'));
hold off;

%% AE
AE = abs(labels - preds);
plot(AE);
title(strcat('MAE = ', num2str(round(mean(AE), 3))));
legend('absolute error')
saveas(gcf, strcat('./', 'plots_', with_K, '_K/', net, '_', with_K, '_K_', 'MAE_', num2str(MAE), '_AE.jpg'));