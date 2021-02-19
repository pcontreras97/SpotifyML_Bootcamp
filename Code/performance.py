import matplotlib.pyplot as plt
import seaborn as sns

def perform_visualisation(params, metrics, ax, param_string):
	acc_train = metrics[0]
	acc_test = metrics[1]
	f1_train = metrics[2]
	f1_test = metrics[3]
	auc = metrics[3]

	sns.lineplot(x = params, y = acc_train, color = "b",
             ax = ax[0])
	sns.lineplot(x = params, y = acc_test, color = "r",
	             ax = ax[0])
	ax[0].set_title("Accuracy Scores changing " + param_string)
	ax[0].set_ylabel("Accuracy")
	ax[0].set_xlabel("Parameter: " + param_string)
	ax[0].legend(["Train", "Test"], loc = "upper right")

	sns.lineplot(x = params, y = f1_train, color = "b",
	             ax = ax[1])
	sns.lineplot(x = params, y = f1_test, color = "r",
	             ax = ax[1])
	ax[1].set_title("F1 Scores changing " + param_string)
	ax[1].set_ylabel("F1 Score")
	ax[1].set_xlabel("Parameter: " + param_string)
	ax[1].legend(["Train", "Test"], loc = "upper right")

	sns.lineplot(x = params, y = auc, color = "r", ax = ax[2])
	ax[2].set_title("Test AUC Scores changing " + param_string)
	ax[2].set_ylabel("Area Under the Curve")
	ax[2].set_xlabel("Parameter: " + param_string);

def plot_roc(roc_dict):
	rocs = list(roc_dict.values())

	for roc in rocs:
	    # roc[0] = fpr, roc[1] = tpr
	    plt.plot(roc[0], roc[1], lw = 2)
	plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
	plt.legend(labels = roc_dict.keys(),
	          loc = "lower right")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.yticks([i/20.0 for i in range(21)])
	plt.xticks([i/20.0 for i in range(21)])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve for Test Set');