library("rhdf5")
library('ggplot2')


sigmoid <- function(x) {
  1 / (1 + exp(-x))
}

get.predictions<-function(x){
  
  pred_class<- names(which.max(x))
  return(pred_class)
  
} 


setwd('~/Cartesius/out_gpu_arch001_002_feat02_lr_bs8_cv_au30/' )

### List attributes
h5ls(file = "epoch_data.hdf5")


### Outputs CNN
pred <- h5read(file = "epoch_data.hdf5", '/epoch_0001/test/outputs')
rownames(pred)<- c(0,1)
prob<- apply(pred,2, sigmoid)
predictions<- as.numeric(apply(prob,2, get.predictions))

### Targets
targets<- h5read(file = "epoch_data.hdf5", '/epoch_0001/test/targets')


### Calulate parameters
# Accuracy
acc<- table(predictions == targets)/161 * 100
print(acc)

# True positive
tp<- length(predictions[predictions == 1 & targets == 1])

# True Negative
tn<- length(predictions[predictions == 0 & targets == 0])

# False Positive
fp<- length(predictions[predictions == 1 & targets == 0])

# False Negative
fn<- length(predictions[predictions == 0 & targets == 1])


#### Plot 
Actual <- factor(c('Xtal', 'Xtal', 'Bio', 'Bio'))
Predicted <- factor(c('Xtal', 'Bio', 'Xtal', 'Bio'))
Y <- c(tn, fp, fn, tp)
Type<- as.factor(c('Correct', 'Wrong', 'Wrong', 'Correct'))
df <- data.frame(Actual, Predicted, Y, Type)
colors<- c('springgreen3', 'firebrick3')
label_size<- 8
base_size<- 20
width<- 8
height<- 5


p <- ggplot(data =  df, mapping = aes(x = Actual, y = Predicted, fill=Type, alpha= 0)) +
  geom_tile() +
  geom_text(aes(label =  Y), vjust = 0.5, fontface  = "bold", size= label_size, alpha= 10) +
  scale_fill_manual(values = colors) +
  theme_bw(base_size) + 
  theme(legend.position = "none",
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title = element_text(face = "bold"),
        axis.text.y = element_text(face="bold"),
        axis.text.x = element_text(face="bold")) + 
  scale_y_discrete(expand = c(0,0)) +
  scale_x_discrete(expand = c(0,0))



ggsave("~/Desktop/confusion_matrix.png", device ="png",plot = p, width = width,height = height, limitsize = F, dpi = 600)







