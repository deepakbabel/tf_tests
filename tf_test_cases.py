import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.core.protobuf import config_pb2

def functionInvocation():
    print("Inside functionInvocation function")
    tf.reset_default_graph()
    g = tf.Graph()
    firstG = g
    with g.as_default():

        def fun1():
            return tf.constant(10)

        def fun2():
            return fun1()
        fun2run = fun2()

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

        def fun1():
            return tf.constant(50)

        fun2run = fun1()

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

        def fun1():
            return tf.multiply(tf.constant(50),tf.constant(20))

        fun2run = fun1()

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

        def fun2(x):
            return fun2(x-1) if x > 1 else tf.constant(x)

        fun2run = fun2(2)

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

        def fun2(x,y):
            retVal = tf.constant(0)
            while x > 0:
                x -= 1
                if x == 5:
                    continue
                print(x)
            for i in range(y):
                i += 1
                if i%8== 0:
                    retVal = tf.constant(i)
                    break
                print(i)
            return retVal

        fun2run = fun2(7,15)

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
        def Forward():
            x = tf.placeholder(tf.float32,shape=(2,2), name="pl1")
            y = tf.placeholder(tf.float32,shape=(2,2), name="pl2")
            return x + y

        data = np.array([[-1, 1], [2, -2]], dtype=np.float32)
        data2 = np.array([[-21, 11], [12, -82]], dtype=np.float32)
        z = Forward()
        feed = {"pl1:0": data, "pl2:0":data2}
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of placeholder run is = ",sess.run(z,feed))

def variables():
    print("Inside variables function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        def Forward(x):
            return tf.Variable(x)

        const = tf.constant(10.0)
        z = Forward(const)
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        print("The output of variables run is = ",sess.run(z))

def doesGraphContainsStatefulOps():
    print("Inside StatefulOps function")
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        def Forward():
            stfn1 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            stfn2 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            stfn3 = tf.ones((2,2),name="fn3")
            return stfn1 + stfn2 + stfn3

        z = Forward()
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
        def Forward():
            with ops.device("/CPU:0"):
                stfn1 = tf.random_uniform((2,2),0.0,5.0,name="fn1")
            with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
                stfn2 = tf.ones((2,2),name="fn3")
            with ops.device("/job:localhost/replica:0/task:0/device:CPU:0"):
                return stfn1 + stfn2

        z = Forward()
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

        def Forward():
            x = tf.constant(35, name='x')
            y = tf.Variable(x + 5, use_resource=True, name='res')
            resVariable = tf.Variable(initial_value=10,use_resource=True);
            refVariable = tf.Variable(initial_value=20,use_resource=False);
            refVariable2 = tf.Variable(initial_value=30);
            a = constant_op.constant([11.0,12], shape=[1, 2])
            b = constant_op.constant([2.0,22.0], shape=[1, 2])
            c = tf.multiply(a, b)
            cvar = tf.Variable(c,name="cvar")
            return y

        z = Forward()

    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(z)
        mylist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        resVarFound,resVarCnt = False,0
        for var in mylist:
            if isResourceVariable(var):
                resVarFound = True
                resVarCnt += 1
        print(resVarFound)
        for var in mylist:
            print(var,var.__class__)
        print("Graph contains so many resource variables",resVarCnt)


if __name__=="__main__":
    functionInvocation()
    constants()
    algrebraicExp()
    recursion()
    controlFlow()
    placeholder()
    variables()
    doesGraphContainsStatefulOps()
    isDeviceAssignmentConsistent()
    doesGraphContainResourceVariables()
