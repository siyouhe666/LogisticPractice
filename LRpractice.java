package aaa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;



/**
 * Logistic实现
 * 此代码仅供交流
 * 为方便理解与查看，本代码全部写与一个java文件中
 * @author siyouhe666@gmail.com
 *
 */
public class LRpractice {
	public static void main(String[] args) {
		double a = 0.009;//学习率
		double[][] instances = fakeIns(2000, 2);//制造2000个假数据，每个数据为2维
		double[] label = fakeLabel(2000);//为假数据提供类标
		int epoch = 10;//迭代次数
		double[] parameters = initialPara(3);//初始化参数
		optimize(instances, label, epoch, parameters, a);//优化过程

	}
	
	
	/**
	 * 创造假数据提供方法训练
	 * 假数据规律：二维数据点，一半处于第一象限内，一半处于第三象限内
	 * @param dim1 样本数
	 * @param dim2 样本维度
	 * @return 假数据矩阵 
	 */
	public static double[][] fakeIns(int dim1, int dim2 ){
		double[][] instances = new double[dim1][dim2];
		for(int i = 0;i<dim1/2;i++){
			for(int j = 0;j<dim2;j++){
				instances[i][j] = Math.random()*10;
			}
		}
		for(int i = dim1/2;i<dim1;i++){
			for(int j = 0;j<dim2;j++){
				instances[i][j] = Math.random()*(-10);
			}
		}
		return instances;
	}
	
	
	/**
	 * 为假数据提供类标
	 * 第一象限：1.0，第三象限0.0
	 * @param dim1 样本数
	 * @return 类标数组
	 */
	public static double[] fakeLabel(int dim1){
		double[] label = new double[dim1];
		for(int i = 0;i<dim1/2;i++){
			label[i] = 1.0;
		}
		for(int i = dim1/2;i<dim1;i++){
			label[i] = 0.;
		}
		return label;
	}
	
	
	/**
	 * 随机初始化参数weight和b
	 * @param dim 参数维度=weight+b
	 * @return parameters:前面为weight最后一维为b
	 */
	public static double[] initialPara(int dim){
		double[] parameters = new double[dim];
		for(int i = 0;i<dim;i++){
			parameters[i] = Math.random();
		}
		return parameters;
	}
	
	/**
	 * sigmoid计算
	 * @param z weight*vector
	 * @return 概率
	 */
	public static double sigmoid(double z){
		return (1/(1+Math.exp(-z)));
	}
	
	/**
	 * 预测过程
	 * @param vec 样本向量
	 * @param weight 参数权重
	 * @param b 偏置
	 * @return z=weight*vector
	 */
	public static double forward(double[] vec, double[] weight, double b){
		assert(vec.length==weight.length);
		double z = 0;
		for(int i = 0;i<vec.length;i++){
			z += vec[i]*weight[i];
		}
		z += b;
		return z;
	}
	
	/**
	 * 对数似然损失函数
	 * y*y^+(1-y)(1-y^)
	 * @param z weight*vector
	 * @param y 真实类标
	 * @return 样本平均损失值
	 */
	public static double loss(double z, double y){
//		System.out.println(sigmoid(z));
		double loss = y*sigmoid(z)+(1-y)*(1-sigmoid(z));
		return loss;
	}
	
	/**
	 * 求w的梯度
	 * @param z
	 * @param y
	 * @param vec
	 * @return gradient_w[]
	 */
	public static double[] backpropation_w(double z, double y, double[] vec){
		double[] gradient = new double[vec.length];
		for(int i = 0;i<vec.length;i++){
			gradient[i] = (y-sigmoid(z))*vec[i];
		}
		return gradient;
	}
	
	/**
	 * b的梯度
	 * @param z
	 * @param y
	 * @return gradient_b
	 */
	public static double backpropation_b(double z, double y){
		double gradient = y-sigmoid(z);
		return gradient;
	}
	
	/**
	 * 生成随机数
	 * @param length 样本数
	 * @return
	 */
	public static List<Integer> shuffle(int length){
		int[] shuff = new int[length];
		List<Integer> list = new ArrayList<Integer>();
		for(int i =0;i<length;i++){
			list.add(i);
		}
		Collections.shuffle(list);
		return list;
	}
	
	/**
	 * 将样本洗牌
	 * @param list 随机顺序
	 * @param instances
	 * @return
	 */
	public static double[][] randomInstances(List<Integer> list, double[][] instances){
		double[][] instancesSort = new double[instances.length][instances[0].length];
		int index = 0;
		for(int i : list){
			instancesSort[index++] = instances[i];
		}
		return instancesSort;
	}
	
	/**
	 * 将类标洗牌
	 * @param list 随机顺序（与样本相对应)
	 * @param label
	 * @return
	 */
	public static double[] randomLabels(List<Integer> list, double[] label){
		double[] labelSort = new double[label.length];
		int index = 0;
		for(int i : list){
			labelSort[index++] = label[i];
		}
		return labelSort;
	}
	
	/**
	 * 梯度求和
	 * @param gw
	 * @param grad_w
	 * @return
	 */
	public static double[] sumGrad_w(double[] gw, double[] grad_w){
		for(int i = 0;i<gw.length;i++){
			grad_w[i] += gw[i];
		}
		return grad_w;
	}
	
	
	/**
	 * 更新参数w
	 * @param weight
	 * @param grad_w
	 * @param a
	 * @return
	 */
	public static double[] updateWeight(double[] weight, double[] grad_w, double a){
		assert(weight.length==grad_w.length);
		for(int i = 0;i<weight.length;i++){
			weight[i] = weight[i] - a*grad_w[i];
		}
		return weight;
	}
	
	/**
	 * 更新参数b
	 * @param b
	 * @param grad_b
	 * @param a
	 * @return
	 */
	public static double updateB(double b, double grad_b, double a){
		b = b - a*grad_b;
		return b;
	}
	
	
	/**
	 * 优化过程
	 * @param instances 样本数据集
	 * @param label 类标
	 * @param epoch 迭代次数
	 * @param parameters 参数：weight+b
	 * @param a 学习率
	 */
	public static void optimize(double[][] instances, double[] label, int epoch, double[] parameters, double a){
		int sampleNum = label.length;//获取样本数
		double[][] instancesSort = null;//每次迭代的随机样本
		double[] labelSort = null;//对应的类标
		double[] weight = new double[parameters.length-1];//weight
		for(int i =0;i<weight.length;i++){
			weight[i] = parameters[i];
		}
		double b = parameters[parameters.length-1];//bias
		for(int epoc = 0;epoc<epoch;epoc++){//迭代
			double[] grad_w = new double[weight.length];
			Arrays.fill(grad_w, 0.);
			double grad_b = 0;
			double loss = 0;
			List<Integer> list = shuffle(sampleNum);
			instancesSort = randomInstances(list, instances);
			labelSort = randomLabels(list, label);
			assert(instancesSort.length==labelSort.length);
			for(int i = 0;i<labelSort.length;i++){
				double[] vec = instancesSort[i];
				double y = labelSort[i];
				double z = forward(vec, weight, b);
				loss += loss(z, y);
				double[] gw = backpropation_w(z, y, vec);
				grad_w = sumGrad_w(gw, grad_w);
				grad_b += backpropation_b(z, y);
			}
			for(int i = 0;i<grad_w.length;i++){
				grad_w[i] /= (double)sampleNum;
			}
			loss /= sampleNum;
			weight = updateWeight(weight, grad_w, a);
			b = updateB(b, grad_b, a);
				System.out.println("------");
				System.out.println("epoch:"+epoc);
				System.out.println("Loss:"+loss);
		}
		
		LogisticParameters lp = new LogisticParameters(weight, b, instances[0].length);
		lp.printInform();
	}
	

/**
 * 保存最后的模型参数
 * @author admin
 *
 */
static class LogisticParameters{
	private double[] weight = null;
	private double b = 0.;
	private int dimNum = -1;
	
	public LogisticParameters(double[] weight, double b, int dimNum){
		this.weight = weight;
		this.b = b;
		this.dimNum = dimNum;
	}
	
	public void printInform(){
		System.out.println();
		System.out.println("Final Parameters");
		System.out.println("***weight***");
		for(double w : weight){
			System.out.print(w+" ");
		}
		System.out.println();
		System.out.println("***bias***");
		System.out.println("b:"+b);
		System.out.println();
	}
}
}


