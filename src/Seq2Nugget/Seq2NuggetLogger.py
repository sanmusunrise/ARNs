from tabulate import tabulate

class Seq2NuggetLogger(object):
    
    def __init__(self):
        self.detection_dev_results = []
        self.detection_test_results = []
        self.all_dev_results = []
        self.all_test_results = []
        
        self.best_dev_detection_epoch = 0
        self.best_test_detection_epoch = 0
        
        self.best_all_test_epoch = 0
        self.best_all_dev_epoch = 0
        
        self.best_boundary_dev_epoch = 0
        self.best_boundary_test_epoch = 0
    
    def best_detection_epoch(self):
        return self.best_dev_detection_epoch
    
    def best_all_epoch(self):
        return self.best_all_dev_epoch
    
    def best_boundary_epoch(self):
        return self.best_boundary_dev_epoch

    def is_best_detection(self,epoch = None):
        if epoch ==None:
            epoch = len(self.detection_dev_results) -1
        is_best_dev = False
        is_best_test = False
        if self.best_dev_detection_epoch == epoch:
            is_best_dev = True
        if self.best_test_detection_epoch == epoch:
            is_best_test = True
        return is_best_dev
        
    def is_best_boundary(self,epoch =None):
        if epoch ==None:
            epoch = len(self.all_dev_results) -1
        is_best_dev = False
        is_best_test = False
        if self.best_boundary_dev_epoch == epoch:
            is_best_dev = True
        if self.best_boundary_test_epoch ==epoch:
            is_best_test = True
        
        return is_best_dev
    
    def is_best_overall(self,epoch = None):
        if epoch ==None:
            epoch = len(self.all_dev_results) -1
        is_best_dev = False
        is_best_test = False
        if self.best_all_dev_epoch == epoch:
            is_best_dev = True
        if self.best_all_test_epoch ==epoch:
            is_best_test = True
        
        return is_best_dev
    

        
    def update_detection_result(self,data_split,p,r,f1):
        p*=100
        r*=100
        f1*=100
        
        if data_split == "dev":
            self.detection_dev_results.append((p,r,f1))
            if f1 >= self.detection_dev_results[self.best_dev_detection_epoch][-1]:
                self.best_dev_detection_epoch = len(self.detection_dev_results) -1
        if data_split =="test":
            self.detection_test_results.append((p,r,f1))
            if f1 >= self.detection_test_results[self.best_test_detection_epoch][-1]:
                self.best_test_detection_epoch = len(self.detection_test_results) -1
                
                
    def update_boundary_result(self,data_split,p,r,f1):
        p*=100
        r*=100
        f1*=100
        
        if data_split == "dev":
            self.all_dev_results.append((p,r,f1))
            if f1 >= self.all_dev_results[self.best_all_dev_epoch][-1]:
                self.best_all_dev_epoch = len(self.all_dev_results) -1
            if self.detection_dev_results[-1][-1] / f1 <= self.detection_dev_results[self.best_boundary_dev_epoch][-1] / self.all_dev_results[self.best_boundary_dev_epoch][-1]:
                self.best_boundary_dev_epoch = len(self.all_dev_results) -1
                
        if data_split =="test":
            self.all_test_results.append((p,r,f1))
            if f1 >= self.all_test_results[self.best_all_test_epoch][-1]:
                self.best_all_test_epoch = len(self.all_test_results) -1
            if self.detection_test_results[-1][-1] / f1 <= self.detection_test_results[self.best_boundary_test_epoch][-1] / self.all_test_results[self.best_boundary_test_epoch][-1]:
                self.best_boundary_test_epoch = len(self.all_test_results) -1
                
    def result_at_epoch(self,epoch = None):
        if epoch == None:
            epoch = len(self.all_dev_results) -1
        rst = []
        
        item = []
        p1,r1,f1 = self.detection_dev_results[epoch]
        item += ["Dev Detection",epoch,p1,r1,f1,self.best_dev_detection_epoch ==epoch]
        rst.append(item)
        
        item = []
        p2,r2,f2 = self.detection_test_results[epoch]
        item += ["Test Detection",epoch,p2,r2,f2,self.best_test_detection_epoch ==epoch]
        rst.append(item)
        
        item = []
        p3,r3,f3 = self.all_dev_results[epoch]
        item += ["Dev All",epoch,p3,r3,f3,self.best_all_dev_epoch ==epoch]
        rst.append(item)
        
        item = []
        p4,r4,f4 = self.all_test_results[epoch]
        item += ["Test All",epoch,p4,r4,f4,self.best_all_test_epoch ==epoch]
        rst.append(item)
        
        rst.append(["-"]*6)
        
        item = []
        item += ["Dev Gap",epoch,p1-p3,r1-r3,f1-f3,self.best_boundary_dev_epoch ==epoch]
        rst.append(item)
        
        item = []
        item += ["Test Gap",epoch,p2-p4,r2-r4,f2-f4,self.best_boundary_test_epoch ==epoch]
        rst.append(item)

        
        return tabulate(rst,headers= ["Metric","Epoch","Precision","Recall","F1","Is_Best"],tablefmt="psql")
