
class DetectionEvaluator(object):

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
            hits = []
            for m in golden[sent_id]:
                hits.append([m,False])

            for t_id,tp in pred[sent_id]:
                find_match = False
                for m in hits:
                    b_id = m[0][0]
                    e_id = m[0][1]
                    m_tp = m[0][2]
                    if t_id >=b_id and t_id < e_id and tp ==m_tp:
                        if not m[1]:
                            m[1] = True
                            find_match = True
                            break
                        find_match = True
                if not find_match:
                    pred_error +=1
            for hit in hits:
                if hit[1]: #and (hit[0][1] - hit[0][0] <= 5):
                    pred_correct+=1

        precision = pred_correct / (pred_correct + pred_error)
        recall = pred_correct / total_mentions
        F1 = 2*(precision * recall) / (precision + recall)

        return precision,recall,F1



