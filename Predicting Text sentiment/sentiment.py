"""
TODO: Answer the questions included in the assignment
"""
import argparse, math, os, re, string, zipfile
from typing import DefaultDict, Generator, Hashable, Iterable, List, Sequence, Tuple
from collections import defaultdict
import numpy as np
from sklearn import metrics
import itertools as iter




class Sentiment:
    """Naive Bayes model for predicting text sentiment"""

    def __init__(self, labels: Iterable[Hashable]):
        """Create a new sentiment model

        Args:
            labels (Iterable[Hashable]): Iterable of potential labels in sorted order.
        """
        self.pos_dictionary = defaultdict(int)
        self.neg_dictionary = defaultdict(int)
        self.total_pos_docs = 0
        self.total_neg_docs = 0
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.sum_pos=0
        self.sum_neg=0
        pass

        

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: Normalized words
        """
        # TODO: Modify the method to generate individual words from the example. Example modifications include
        # removing punctuation and/or normalizing case (e.g., making all lower case)

        example = example.translate(str.maketrans('', '', string.punctuation))
        
        return example.lower().split()

    def add_example(self, example: str, label: Hashable, id:str = None):
        """Add a single training example with label to the model

        Args:
            example (str): Text input
            label (Hashable): Example label
            id (str, optional): File name from training/test data (may not be available). Defaults to None.
        """
        # TODO: Implement function to update the model with words identified in this training example
        pass

        normalized_words = self.preprocess(example)

        if label ==0:
            self.total_neg_docs +=1
            for word in normalized_words:
                self.total_neg_words +=1
                if word in self.neg_dictionary:
                    self.neg_dictionary[word]+=1

                else:
                    self.neg_dictionary[word] = 1
        
        if label ==1:
            self.total_pos_docs +=1
            for word in normalized_words:
                self.total_pos_words+=1
                if word in self.pos_dictionary:
                    self.pos_dictionary[word]+=1
                else:
                    self.pos_dictionary[word]=1

    def predict(self, example: str, pseudo=0.0001, id:str = None) -> Sequence[float]:
        """Predict the P(label|example) for example text, return probabilities as a sequence

        Args:
            example (str): Test input
            pseudo (float, optional): Pseudo-count for Laplace smoothing. Defaults to 0.0001.
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            Sequence[float]: Probabilities in order of originally provided labels

        print(self.total_pos_docs) 
        print(self.total_neg_docs) 
        print(self.total_pos_words) 
        print(self.total_neg_words)
        print(self.sum_pos)
        print(self.sum_neg)
        """

        

        total_pos = np.log(self.total_pos_docs/(self.total_pos_docs + self.total_neg_docs))
        total_neg = np.log(self.total_neg_docs/(self.total_pos_docs + self.total_neg_docs))

        positive =[]
        negative = []


        for word in self.preprocess(example):
            num_pos = self.pos_dictionary[word]
            num_neg = self.neg_dictionary[word]

            p_word_pos = (num_pos+pseudo)/(self.total_pos_words + len(self.pos_dictionary)*pseudo)
            p_word_neg = (num_neg+pseudo)/(self.total_neg_words + len(self.neg_dictionary)*pseudo)

            positive.append(p_word_pos)
            negative.append(p_word_neg)
            
        total_pos += sum(np.log(positive))
        total_neg += sum(np.log(negative))
        total_dnm = np.logaddexp.reduce([total_pos, total_neg])

        total_pos = np.exp(total_pos-total_dnm) 
        total_neg = np.exp(total_neg-total_dnm) 
        
        return [total_neg, total_pos]

class CustomSentiment(Sentiment):
    # TODO: Implement your custom Naive Bayes model
    def __init__(self, labels: Iterable[Hashable]):
        super().__init__(labels)

    def preprocess(self, example: str, id:str =None) -> List[str]:
        """Normalize the string into a list of words.

        Args:
            example (str): Text input to split and normalize
            id (str, optional): File name from training/test data (may not be available). Defaults to None.

        Returns:
            List[str]: Normalized words
        """
        # TODO: Modify the method to generate individual words from the example. Example modifications include
        # removing punctuation and/or normalizing case (e.g., making all lower case)
        total_nmer=[]
        def gen_nmer (word, n_mer):
            nmers=[]
            if len(word)>n_mer:
                for i in range(len(word)-(n_mer-1)):
                    nmers.append(word[i:i+n_mer])
            return nmers
        
        example = example.translate(str.maketrans('', '', string.punctuation))
        
        example = example.lower().split()

        #print(example+total_nmer)
        for word in example:
            for i in range(2,6):
                total_nmer+=gen_nmer(word,i)

        for i in range(len(example)-2):
            total_nmer.append(example[i]+" " + example[i+1])
            total_nmer.append(example[i]+" " + example[i+1]+ " " + example[i+2])
        
        if len(example)==2:
    
            total_nmer.append(example[-2] + " " + example[-1])
        
        return example+total_nmer
        
def process_zipfile(filename: str) -> Generator[Tuple[str, str, int], None, None]:
    """Create generator of labeled examples from a Zip file that yields a tuple with
    the id (filename of input), text snippet and label (0 or 1 for negative and positive respectively).

    You can use the generator as a loop sequence, e.g.

    for id, example, label in process_zipfile("test.zip"):
        # Do something with example and label

    Args:
        filename (str): Name of zip file to extract examples from

    Yields:
        Generator[Tuple[str, str, int], None, None]: Tuple of (id, example, label)
    """
    with zipfile.ZipFile(filename) as zip:
        for info in zip.infolist():
            # Iterate through all file entries in the zip file, picking out just those with specific ratings
            match = re.fullmatch(r"[^-]+-(\d)-\d+.txt", os.path.basename(info.filename))
            if not match or (match[1] != "1" and match[1] != "5"):
                # Ignore all but 1 or 5 ratings
                continue
            # Extract just the relevant file the Zip archive and yield a tuple
            with zip.open(info.filename) as file:
                yield (
                    match[0],
                    file.read().decode("utf-8", "ignore"),
                    1 if match[1] == "5" else 0,
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
    parser = argparse.ArgumentParser(description="Train Naive Bayes sentiment analyzer")

    parser.add_argument(
        "--train",
        default="data/train.zip",
        help="Path to zip file or directory containing training files.",
    )
    parser.add_argument(
        "--test",
        default="data/test.zip",
        help="Path to zip file or directory containing testing files.",
    )
    parser.add_argument(
        "-m", "--model", default="base", help="Model to use: One of base or custom"
    )
    parser.add_argument("example", nargs="?", default=None)

    args = parser.parse_args()

    # Train model
    if args.model == "custom":
        model = CustomSentiment(labels=[0, 1])
    else:
        model = Sentiment(labels=[0, 1])
    for id, example, y_true in process_zipfile(
        os.path.join(os.path.dirname(__file__), args.train)
    ):
        model.add_example(example, y_true, id=id)

    # If interactive example provided, compute sentiment for that example
    if args.example:
        print(model.predict(args.example))
    else:
        predictions = []
        for id, example, y_true in process_zipfile(
            os.path.join(os.path.dirname(__file__), args.test)
        ):
            # Determine the most likely class from predicted probabilities
            #print(id, example, model.predict(example,id=id))
            predictions.append((id, y_true, np.argmax(model.predict(example,id=id))))
            #if y_true != predictions[-1][2]:
                #print (example, y_true, predictions[-1][2])

        # Compute and print accuracy metrics
        _, y_test, y_true = zip(*predictions)
        predict_metrics = compute_metrics(y_test, y_true)
        for met, val in predict_metrics.items():
            print(
                f"{met.capitalize()}: ",
                ("\n" if isinstance(val, np.ndarray) else ""),
                val,
                sep="",
            )

