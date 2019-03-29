def Train(Model,Data_Loader,Opt,Epoch,Criterion):
    Model
    Model.train() # Put the model in training mode
    for batch in tqdm(Data_Loader):
        inputs=batch['X'].long()
        labels=batch['y'].long()
        outputs=Model(inputs)
        Opt.zero_grad()
        loss=Criterion(outputs,labels)
        loss.backward()
        Opt.step()
    print("Epoch {}: Loss {}".format(Epoch+1,loss.item()))    

def Test(Model,Data_Loader):
    Model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for batch in Data_Loader:
            inputs=batch['X'].long()
            labels=batch['y'].long()
            outputs=Model(inputs)
            test_loss=test_loss+F.cross_entropy(outputs,labels,reduction='sum').item() #sum up batch loss
            pred=outputs.argmax(dim=1,keepdim=True)
            correct=correct+pred.eq(labels.view_as(pred)).sum().item()
    test_loss=test_loss/len(Data_Loader.dataset)
    accuracy=(correct/len(Data_Loader.dataset))*100.0
    print("-"*10)
    print("Test loss {}, Test accuracy {}/{} {}".format(test_loss,correct,len(Data_Loader.dataset),accuracy))    