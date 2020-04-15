from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import function
from tensorflow.python.ops import gen_functional_ops
import tensorflow as tf
import numpy as np

def functionInvocation():
    print("Inside functionInvocation function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        def fun1(x,y):
            return tf.multiply(tf.constant(10),(x+y))

        @function.Defun(tf.int32,tf.int32,func_name="fun2Fn")
        def fun2(x,y):
            return fun1(x,y)

        fun2run = gen_functional_ops.StatefulPartitionedCall(args=[tf.constant(10),tf.constant(20)],Tout=[tf.int32],f=fun2)

    with tf.Session(graph=g) as sess: #If you do not pass the graph here, the session would not know about 'g' graph operations/funcs
        print("sess is in scope of 'g' graph ") if(g is firstG) else print("sess is not in scope of 'g' graph")
        sess.run(tf.global_variables_initializer())
        out = sess.run(fun2run)
        print("Function Invocation output is = ",out)

def constants():
    print("Inside constants function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        @function.Defun(*[tf.int32]*2,func_name="fun1Fn")
        def fun1(x,y):
            return tf.multiply(x,y)

        # fun2run = fun1()
        arg1 = tf.constant( [ [1,2], [3,4] ] )
        arg2 = tf.constant( [ [2,2], [4,5] ] )
        fun2run = gen_functional_ops.StatefulPartitionedCall(args=[ arg1, arg2 ], Tout=[tf.int32],f=fun1)

    with tf.Session(graph=g) as sess: #If you do not pass the graph here, the session would not know about 'g' graph operations/funcs
        print("sess is in scope of 'g' graph ") if(g is firstG) else print("sess is not in scope of 'g' graph")
        sess.run(tf.global_variables_initializer())
        out = sess.run(fun2run)
        print("constants invocation output is = ",out)

def algrebraicExp():
    print("Inside algrebraicExp function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        @function.Defun(*[tf.int32]*2,func_name="fun1Fn")
        def fun1(x,y):
            temp = tf.add(x,x)
            z = tf.multiply(temp,y)
            return z

        arg1 = tf.constant(10)
        arg2 = tf.constant([20])
        fun2run = gen_functional_ops.StatefulPartitionedCall(args=[arg1, arg2], Tout=[tf.int32],f=fun1)

    with tf.Session(graph=g) as sess: #If you do not pass the graph here, the session would not know about 'g' graph operations/funcs
        print("sess is in scope of 'g' graph ") if(g is firstG) else print("sess is not in scope of 'g' graph")
        sess.run(tf.global_variables_initializer())
        out = sess.run(fun2run)
        print("Algebraic invocation output is = ",out)

def recursion():
    print("Inside recursion function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        def body(x):
            a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
            b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
            c = a + b
            return tf.nn.relu(x + c)

        def condition(x):
            return tf.reduce_sum(x) < 100

        @function.Defun(*[tf.int32])
        def fun(arg):
            x = tf.Variable(tf.constant(arg, shape=[2, 2]))
            return tf.constant(2)
            # tf.while_loop(condition,body,[x])

        arg1 = tf.constant([1,1],[3,3])
        fun2run = gen_functional_ops.StatefulPartitionedCall(args=[arg1],Tout=[tf.int32],f=fun)

    with tf.Session(graph=g) as sess: #If you do not pass the graph here, the session would not know about 'g' graph operations/funcs
        print("sess is in scope of 'g' graph ") if(g is firstG) else print("sess is not in scope of 'g' graph")
        sess.run(tf.global_variables_initializer())
        out = sess.run(fun2run)
        print("recursion output is = ",out)

def controlFlow():
    print("Inside controlFlow function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        @function.Defun(tf.int32,tf.int32)
        def fun2(x,y):
            val = 10
            for i in range(20):
                i += 1
                if i%8== 0:
                    retVal = tf.constant(i)
                    break
            return retVal

        arg1 = tf.constant(7)
        arg2 = tf.constant(15)
        fun2run = gen_functional_ops.StatefulPartitionedCall(args=[7, 15], Tout=[tf.int32],f=fun2)

    with tf.Session(graph=g) as sess: #If you do not pass the graph here, the session would not know about 'g' graph operations/funcs
        print("sess is in scope of 'g' graph ") if(g is firstG) else print("sess is not in scope of 'g' graph")
        sess.run(tf.global_variables_initializer())
        out = sess.run(fun2run)
        print("control flow output is = ",out)


def placeholder():
    print("Inside placeholder function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        @function.Defun(*[tf.int32]*2)
        def Forward(x,y):
            #Do not create placeholders in Defun methods..placeholders should be created outside of Defun()..and can be passed inside it
            print(x.name)
            print(y.name)
            b = tf.add(x, y)
            return b
        pl1 = tf.placeholder(tf.int32,name="pl1")
        pl2 = tf.placeholder(tf.int32,name="pl2")
        data = np.array([[-1, 1], [2, -2]], dtype=np.int32)
        data2 = np.array([[-2, 3], [4, -6]], dtype=np.int32)
        z = gen_functional_ops.StatefulPartitionedCall(args=[pl1,pl2], Tout=[tf.int32],f=Forward)

        feed = {"pl1:0": data,"pl2:0": data2}
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of placeholder run is = ",sess.run(z,feed))

def variables():
    print("Inside variables function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        @function.Defun(tf.int32,tf.int32)
        def Forward(x,y):
            #create variables outside Defun() method, you can pass variables inside Defun method though
            return tf.multiply(x,y)
        const1 = tf.constant(10)
        const2 = tf.constant(20)
        var1 = tf.Variable(const1, dtype=tf.int32)
        var2 = tf.Variable(const2, dtype=tf.int32)

        z = gen_functional_ops.StatefulPartitionedCall(args=[var1,var2],Tout=[tf.int32], f=Forward)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of variables run is = ",sess.run(z))

def doesGraphContainsStatefulOps():
    print("Inside StatefulOps function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        @function.Defun()
        def Forward():
            stfn1 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            stfn2 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            stfn3 = tf.ones((2,2),name="fn3")
            return stfn1 + stfn2 + stfn3

        z = gen_functional_ops.StatefulPartitionedCall(args=[], Tout=[tf.float32],f=Forward)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of stateful ops run is = ",sess.run(z))
        stateful_ops = [(op.name, op.type) for op in sess.graph.get_operations() if op.op_def.is_stateful]
        print("Graph contains {0} stateful_ops: ",len(stateful_ops))

def isDeviceAssignmentConsistent():
    print("Inside device assignment function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        @function.Defun()
        def Forward():
            with ops.device("/CPU:0"):
                stfn1 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
                stfn2 = tf.ones((2,2),name="fn3")
            with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
                return stfn1 + stfn2

        z = gen_functional_ops.StatefulPartitionedCall(args=[], Tout=[tf.float32],f=Forward)
    run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    with tf.Session(graph=g,config=config_pb2.ConfigProto(device_count={"CPU": 1})) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of device assignment run is = ",sess.run(z, options=run_options,run_metadata=run_metadata))
        assignedDevicesSet = set()
        for func in run_metadata.step_stats.dev_stats:
            print("device used: ", repr(func.device))
            assignedDevicesSet.add(func.device)
        print ("Device assignment inconsistent") if len(assignedDevicesSet) > 2 else print("Device assignment is consistent")

refVar = tf.Variable(20)
resVar = tf.Variable(20,use_resource=True)
def isResourceVariable(var):
    refVariableClass = refVar.__class__
    return bool(not issubclass(var.__class__,refVariableClass) and issubclass(var.__class__,resVar.__class__))

def doesGraphContainResourceVariables():
    mylist = None
    tf.reset_default_graph()
    g = ops.Graph()
    with g.as_default():

        @function.Defun(tf.int32)
        def Forward(x):
            a = constant_op.constant(2)
            c = tf.multiply(x, a)
            return c

        x = tf.constant(35, name='x')
        resVar = tf.Variable(x + 5, use_resource=True, name='res')
        z = gen_functional_ops.StatefulPartitionedCall(args=[resVar], Tout=[tf.int32],f=Forward)

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(z)
        mylist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        resVarFound = False
        for var in mylist:
            if isResourceVariable(var):
                print("Found a resource variable")
                resVarFound = True
                # break
            else:
                continue
        print(resVarFound)
        for var in mylist:
            print(var,var.__class__)
        resVarCnt = len(mylist)
        print("Graph contains",resVarCnt, "number of resource variables")


if __name__=="__main__":
    functionInvocation()
    constants()
    algrebraicExp()
    # recursion() - doesn't work
    controlFlow()
    placeholder()
    variables()
    doesGraphContainsStatefulOps()
    isDeviceAssignmentConsistent()
    doesGraphContainResourceVariables()
