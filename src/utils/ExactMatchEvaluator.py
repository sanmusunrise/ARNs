
class ExactMatchEvaluator(object):

    def __init__(self):
        pass

    def eval(self,pred,golden):
        
        total_mentions = 0.0
        pred_error = 0.0
        pred_correct =  0.0
        for sent_id in golden:
            total_mentions += len(golden[sent_id])
            if not sent_id in pred:
                continue
            rst = set()
            for b,e,tp,_ in pred[sent_id]:
                rst.add((b,e,tp))
                #print sent_id,b,e,tp
            for b,e,tp in rst:
                if (b,e,tp) in golden[sent_id]:
                    pred_correct +=1
                else:
                    pred_error +=1
        #print pred_correct,pred_error,total_mentions
        if pred_correct ==0:
            return 0.0,0.0,0.0
        precision = pred_correct / (pred_correct + pred_error)
        recall = pred_correct / total_mentions
        F1 = 2*(precision * recall) / (precision + recall)

        return precision,recall,F1



