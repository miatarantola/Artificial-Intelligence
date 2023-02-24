"""
Name:Mia Tarantola
Class: CS311: AI
Assignment: PA_3 Decision Trees

_________________________________________________________________________________________________
HEPATITIS DATA SET
The hepatitis data set performs farely well with a 0.87 accuracy rate, meaning it predicts
the correct label 87% of the time. The confiusion matrix results in 7 true negatives,
1 false positive, 4 false negatives and 38 true positive which is also pretty good!
We also have a 90% true positiv rate (recall) and a 93% precision rate.

The tree is interesting but what I had expected. Becuase varices are complication of late 
stage hepatitis it makes sense that most of the tree is contained in the varices = yes branch.

_________________________________________________________________________________________________
ADULT DATA SET

features: age, education, marital-status, relationship, capital-difference, hours-per-week

I chose these attributes as these were the ones with the most clear splitting. Most of the unique 
values for these attributes were not very split (mostly 1 or mostly 0), so these made for good options. 
I also picked my discretization based on the histograms provided. It also helped to combine some of the 
categorical varibles (ie. education). If there are multiple values that have similar label distributions
it may help to combine those values into one category to reduce the amount of splitting needed.

Our model can be applied to a credit card the provides cash back to high earners. If we are given
information about applicants, our tree can help predict whether those applicants are likely to make more
than/ less than 50k per year.

"""


#________________________________________________________________________________________________
#IMPORTS
import argparse, os, random, sys
from math import remainder
from typing import Any, Dict, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
#___________________________________________________________________________________________________

# Type alias for nodes in decision tree
DecisionNode = Union["DecisionBranch", "DecisionLeaf"]


class DecisionBranch:
    """Branching node in decision tree"""

    def __init__(self, attr: str, branches: Dict[Any, DecisionNode]):
        """Create branching node in decision tree

        Args:
            attr (str): Splitting attribute
            branches (Dict[Any, DecisionNode]): Children nodes for each possible value of `attr`
        """
        self.attr = attr
        self.branches = branches

    def predict(self, x: pd.Series):
        """Return predicted label for array-like example x"""
        # TODO: Implement prediction based on value of self.attr in x
        subtree = self.branches[x[self.attr]]
        return subtree.predict(x)


    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Test Feature", self.attr)
        for val, subtree in self.branches.items():
            print(" " * 4 * indent, self.attr, "=", val, "->", end=" ")
            subtree.display(indent + 1)


class DecisionLeaf:
    """Leaf node in decision tree"""

    def __init__(self, label):
        """Create leaf node in decision tree

        Args:
            label: Label for this node
        """
        self.label = label

    def predict(self, x):
        """Return predicted labeled for array-like example x"""
        return self.label

    def display(self, indent=0):
        """Pretty print tree starting at optional indent"""
        print("Label=", self.label)




def information_gain(X: pd.DataFrame, y: pd.Series, attr: str) -> float:
    """Return the expected reduction in entropy from splitting X,y by attr"""
    # TODO: Implement information gain metric for selecting attributes

    def B(prob):
        if prob ==0:
            return -(1-prob)*np.log2(1-prob)
        elif prob ==1:
            return -(prob*np.log(prob))
        else:
            b_prob = -(prob*np.log2(prob)+(1-prob)*np.log2(1-prob))
            return b_prob
    
    y_test = pd.Series(y)
    num_examples = len(X.index) #how many examples are in current data set 
    prev = y_test.value_counts().transform(lambda x: x/x.sum()) #prev prob of ones/total
    prev_b = B(prev.loc[1]) #parent B

    counts = X.groupby([attr,y]).size() #splits unique values in attr and counts of 1s and 0s

    unique_attrs = X[attr].unique() #list of unique values for attr    
    total_gain = 0 

    for i in unique_attrs:
        num_ones = counts.loc[i,1] #number of ones
        num_zeros = counts.loc[i,0] #number of zeros
        curr_gain = ((num_ones+num_zeros)/num_examples) * B(float(num_ones/(num_ones+num_zeros))) 
        total_gain+=curr_gain
    return prev_b - total_gain 
    


def learn_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    attrs: Sequence[str],
    y_parent: pd.DataFrame,
) -> DecisionNode:
    """Recursively learn the decision tree

    Args:
        X (pd.DataFrame): Table of examples (as DataFrame)
        y (pd.Series): array-like example labels (target values)
        attrs (Sequence[str]): Possible attributes to split examples
        y_parent (pd.Series): array-like example labels for parents (parent target values)

    Returns:
        DecisionNode: Learned decision tree node
    """

    # TODO: Implement recursive tree construction based on pseudo code in class
    # and the assignment

    #BASE CASE1
    if X.empty :
        return DecisionLeaf(y_parent.mode()[0])

    #BASE CASE 2    
    elif len(y.unique())==1:
        return DecisionLeaf(y.unique()[0])

    #BASE CASE3    
    elif len(attrs)==0:
        return DecisionLeaf(y.mode()[0])
    else:

        #find attr with highesst information gain
        highest_gain = 0
        highest_attr=attrs[0]
        
        for a in attrs:

            gain = information_gain(X,y,a)
            if gain > highest_gain:
                highest_gain = gain
                highest_attr = a
       
  

        #create branch dict
        branch_dict = dict()
        tree = DecisionBranch(highest_attr, branch_dict)

        #delete splitting attr
        attrs = np.delete(attrs,np.where(attrs.values == highest_attr))

        for v, new_examples in X.groupby(highest_attr):
            retained_indices = list(np.where(X[highest_attr]==v)[0]) #list of y pos. to keep
            new_y = y.iloc[retained_indices] #new y values
            subtree = learn_decision_tree(new_examples,new_y,attrs,y) #create subtree
            branch_dict[v]=subtree #add to dict
        

    return tree


def fit(X: pd.DataFrame, y: pd.Series) -> DecisionBranch:
    """Return train decision tree on examples, X, with labels, y"""
    # You can change the implementation of this function, but do not modify the signature
    return learn_decision_tree(X, y, X.columns, y)


def predict(tree: DecisionNode, X: pd.DataFrame):
    """Return array-like predctions for examples, X and Decision Tree, tree"""

    # You can change the implementation of this function, but do not modify the signature

    # Invoke prediction method on every row in dataframe. `lambda` creates an anonymous function
    # with the specified arguments (in this case a row). The axis argument specifies that the function
    # should be applied to all rows.


    return X.apply(lambda row: tree.predict(row), axis=1)


def load_adult(feature_file: str, label_file: str):

    # Load the feature file
    examples = pd.read_table(
        feature_file,
        dtype={
            "age": int,
            "workclass": "category",
            "education": "category",
            "marital-status": "category",
            "occupation": "category",
            "relationship": "category",
            "race": "category",
            "sex": "category",
            "capital-gain": int,
            "capital-loss": int,
            "hours-per-week": int,
            "native-country": "category",
        },
    )
    labels = pd.read_table(label_file).squeeze().rename("label")



    # TODO: Select columns and choose a discretization for any continuos columns. Our decision tree algorithm
    # only supports discretized features and so any continuous columns (those not already "category") will need
    # to be discretized.

    # For example the following discretizes "hours-per-week" into "part-time" [0,40) hours and
    # "full-time" 40+ hours. Then returns a data table with just "education" and "hours-per-week" features.

    examples["hours-per-week"] = pd.cut(
        examples["hours-per-week"],
        bins= [0,31,41,101],
        right=False,
        labels=["<30","31-40","41-100"],
    )


    examples["age"] = pd.cut(
        examples["age"],
        bins = [0,25,45,65,100],
        right = False,
        labels = ["young", "mid","senior","older"],

    )

    examples['capital-difference'] = examples['capital-gain'] - examples['capital-loss']

    examples["capital-gain"] = pd.cut(
        examples["capital-gain"],
        bins=[0,8001,sys.maxsize],
        right = False,
        labels = ["0 - 8000","8000<",],
    )

    

    examples["capital-loss"] = pd.cut(
        examples["capital-loss"],
        bins=[0,301,sys.maxsize],
        right = False,
        labels = ["0 - 300","300+"],
    )
    examples["capital-difference"] = pd.cut(
        examples["capital-difference"],
        bins = [-5000,5000,sys.maxsize],
        right = False,
        labels = ["minor loss/gain","major gain"],

    )

    #combine lower levels as all are = 0
    examples['education'].replace(['Preschool', '1st-4th','5th-6th', '7th-8th', '9th' ,'10th','11th',  '12th'],
                             ' Pre-k to High School Graduate', inplace = True)

    examples['marital-status'] = examples['marital-status'].replace([ 'Separated','Widowed',  'Never-married','Divorced','Married-spouse-absent'], 'single')

    examples['marital-status'] = examples['marital-status'].replace([ 'Married-civ-spouse', 'Married-AF-spouse'], 'married')
    examples['workclass'] = examples['workclass'].replace(['Self-emp-not-inc','Local-gov','State-gov','Self-emp-inc','Federal-gov','Without-pay'],'non-private sector')

    races = list(examples["race"].unique())

    #examples['race'] = examples['race'].replace(races,'other')
    return examples[["age","education","marital-status","relationship","capital-difference","hours-per-week"]],labels

# You should not need to modify anything below here


def load_examples(
    feature_file: str, label_file: str, **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load example features and labels. Additional arguments are passed to
    the pandas.read_table function.

    Args:
        feature_file (str): Delimited file of categorical features
        label_file (str): Single column binary labels. Column name will be renamed to "label".

    Returns:
        Tuple[pd.DataFrame,pd.Series]: Tuple of features and labels
    """
    return (
        pd.read_table(feature_file, dtype="category", **kwargs),
        pd.read_table(label_file, **kwargs).squeeze().rename("label"),
    )


def compute_metrics(y_true, y_pred):
    """Compute metrics to evaluate binary classification accuracy

    Args:
        y_true: Array-like ground truth (correct) target values.
        y_pred: Array-like estimated targets as returned by a classifier.

    Returns:
        dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
    """
   
    return {
        "confusion": metrics.confusion_matrix(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test decision tree learner")
    parser.add_argument(
        "-p",
        "--prefix",
        default="small1",
        help="Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, hepatitis, adult.",
    )
    parser.add_argument(
        "-k",
        "--k_splits",
        default=10,
        type=int,
        help="Number of splits for stratified k-fold testing",
    )


    args = parser.parse_args()

    if args.prefix != "adult":
        # Derive input files names for test sets
        train_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_data.txt"
        )
        train_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.train_label.txt"
        )
        test_data_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_data.txt"
        )
        test_labels_file = os.path.join(
            os.path.dirname(__file__), "data", f"{args.prefix}.test_label.txt"
        )

        # Load training data and learn decision tree
        train_data, train_labels = load_examples(train_data_file, train_labels_file)
        tree = fit(train_data, train_labels)
        tree.display()

        # Load test data and predict labels with previously learned tree
        test_data, test_labels = load_examples(test_data_file, test_labels_file)
        pred_labels = predict(tree, test_data)

        # Compute and print accuracy metrics
        predict_metrics = compute_metrics(test_labels, pred_labels)
        for met, val in predict_metrics.items():
            print(
                met.capitalize(),
                ": ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )
    else:
        # We use a slightly different procedure with "adult". Instead of using a fixed split, we split
        # the data k-ways (preserving the ratio of output classes) and test each split with a Decision
        # Tree trained on the other k-1 splits.
        data_file = os.path.join(os.path.dirname(__file__), "data", "adult.data.txt")
        labels_file = os.path.join(os.path.dirname(__file__), "data", "adult.label.txt")
        data, labels = load_adult(data_file, labels_file)

        scores = []

        kfold = StratifiedKFold(n_splits=args.k_splits)
        i = 1
        for train_index, test_index in kfold.split(data, labels):
            print(i)
  
            X_train, X_test = data.iloc[train_index], data.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

            tree = fit(X_train, y_train)
            y_pred = predict(tree, X_test)

            
            
            scores.append(metrics.accuracy_score(y_test, y_pred))
            tree.display()
            i+=1
            

        print(
            f"Mean (std) Accuracy (for k={kfold.n_splits} splits): {np.mean(scores)} ({np.std(scores)})"
        )