import numpy as np 
import tensorflow as tf 

import reader

DATA_PATH = 'C:\\Users\\Administrator\\Desktop\\data'
HIDDEN_SIZE = 200 # 隐藏层规模
NUM_LAYERS = 2 # 深层循环神经网络中 LSTM 的层数
VOCAB_SIZE = 10000 # 词典规模，总共一万个单词
 
LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 30
TRAIN_NUM_STEP = 35  # 训练数据截断长度,也就是上面的 5 ，也是 RNN 的最大时间序列长度
 
# 在测试时不需要使用截断，所以可以将测试数据看成一个超长的序列
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
 
NUM_EPOCH = 2
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5 # 用于控制梯度膨胀的参数
class PTBModel(object):
    def __init__(self,is_training,batch_size,num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
 
        # 定义输入层。输入层的维度为 batch_size * num_steps
        self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])
        
        # 定义预期输出，它的维度和输入是一样的
        self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])
        
        # 定义lstm 的dropout 结构
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training :
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell,output_keep_prob = KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
 
        # 初始化状态，也就是全零的向量
        self.initial_state = cell.zero_state(batch_size,tf.float32)
        # 将单词ID 转换成单词向量，因为总共有 VOCAB_SIZE 个单词，每个单词的向量维度设置成
        # HIDDEN_SIZE,所以 embedding 参数的维度为 VOCAB_SIZE * HIDDEN_SIZE
        embedding = tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])
        
        # 将原本 batch_size * num_steps 个单词ID 转换成词向量
        # 变成了三维 batch_size * num_steps * HIDDEN_SIZE 的数据
        inputs = tf.nn.embedding_lookup(embedding,self.input_data)
 
        if is_training: inputs = tf.nn.dropout(inputs,KEEP_PROB)
        
        # 定义输出列表。先将不同时刻的 LSTM 结构的输出收集起来，再通过全连接层得到最终的输出
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step >0 :tf.get_variable_scope().reuse_variables()
                ''' 循环神经网络 lstm 结构的标准输入包括 batch 的话是一个二维的数组 
                    不包括batch 在内就是一维数组，跟全连接神经网络输入结构一样 '''
                cell_output,state = cell(inputs[:,time_step,:],state)
                # 输出应该就是一个二维数组 batch_size * HIDDEN_SIZE
                outputs.append(cell_output)
        # reshape 成 [batch_size * num_steps, HIDDEN_SIZE] 形状
        output = tf.reshape(tf.concat(outputs,1),[-1,HIDDEN_SIZE])
 
        # 将从 lstm 中得到的输出再经过一个全连接层得到最后的预测结果
        # 最终的预测结果在每一个时刻上都是一个长度为 VOCAB_SIZE 的数组
        # 经过 softmax 层表示成下一个单词的概率
        # 结果为 [batch_size * num_steps,VOCAB_SIZE] 数组
        weight = tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
        bias = tf.get_variable("bias",[VOCAB_SIZE])
        logits = tf.matmul(output,weight) + bias
        
        # 定义交叉熵损失函数， TensorFlow 提供了 sequence_loss_by_example 
        # 计算一个序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self.targets,[-1])], # 将[batch_size ，num_steps]压缩成一维
                
                # 损失的权重 所有的权重都为1 ，意思是不同的batch ，不同的时刻权重一样
                [tf.ones([batch_size * num_steps],dtype = tf.float32)])
        # 计算每个 batch 的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
 
        # 只在训练模型时定义反向传播操作，否则到此为止
        if not is_training: return 
        trainable_variables = tf.trainable_variables()
        # 通过 clip_by_global_norm 函数控制梯度大小，避免梯度膨胀问题
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost,trainable_variables),MAX_GRAD_NORM)
        
        # 定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
 
def run_epoch(session,model,data,train_op,output_log):
    # 计算 perplexity 辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 使用当前数据训练或者测试模型
    for step,(x,y) in enumerate(
            reader.ptb_producer(data,model.batch_size,model.num_steps)):
        print(step,111111111)
        print((x,y),2222222222222)
        cost,state,_ = session.run(
                      [model.cost,model.final_state,train_op],
                      {model.input_data:x,model.targets:y,model.initial_state:state})
        total_costs += cost
        iters += model.num_steps
        
        # 只有在训练时输出日志
        if output_log and step % 100 == 0:
            print ("After %d steps,perplexity is %.3f" \
                            %(step,np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)
 
def main(_):
    # 获取数据
    train_data,valid_data,test_data,_ = reader.ptb_raw_data(DATA_PATH)
    
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05,0.05)
    # 定义训练时的循环神经网络模型
    with tf.variable_scope("language_model",initializer = initializer):
        train_model = PTBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)
    
    # 定义评测用的循环神经网络模型
    with tf.variable_scope("language_model",reuse = True,initializer = initializer):
        eval_model = PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)
    
    with tf.Session() as session:
        tf.initialize_all_variables().run()
        for i in range(NUM_EPOCH):
            print ("In iteration: %d" % (i+1))
            run_epoch(session,train_model,train_data,train_model.train_op,True)
            
            # 使用验证集测试效果
            valid_perplexity = run_epoch(
                        session,eval_model,valid_data,tf.no_op(),Fasle)
            print("Eopch: %d Validation Perplexity :%.3f" %(i+1,valid_perplexity))
 
if __name__ == "__main__":
    tf.app.run()