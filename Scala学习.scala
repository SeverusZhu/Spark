Scala

println()  // 可以输出，计算，和进行字符串的拼接输出

值：
val x = 1 + 1   // val numbers = Array(1,2,1,4,5)   有序     val numbers = List(1,2,3,4,1,2,3)  有序 ，不可变   val numbers = Set(1,2,3,1,2,3)  无序，不可变        
println(x)      // numbers(3) = 10         数组Array可变

值无法被重新分配. 如上述的后面 添加 x = 3 则会报错; 但是这个可以修改为
var x = 1 + 1
x = 3

元组Tuple：
元组在不使用类的情况下，将元素组合起来形成简单的逻辑集合
val hostPort = ("localhost", 80)     // 元组不能通过名称获取字段，只能使用位置下标，而且下标基于1
hostPort._1 // String = localhost  
hostPort._2 // Int = 80 

1 -> 2 // (Int, Int) = (1,2)  创建二元数组


显式声明类型： 
val x: Int = 1 + 1
var x: Int = 1 + 1

块：
println(
	val x = 1 + 1
	x + 1
})

匿名函数: (x: Int) => x + 1
命名函数： val getTheAnswer = () => 42				方法： def name: String = System.getProperty("user.name")
		  println(getTheAnswer())						  println("Hello, " + name + "!")

		  val addOne = (x: Int) => x + 1				  def add(x: Int, y: Int): Int = x + y
		  println(addOne(1))							  println(add(1, 2))

		  val add = (x: Int, y: Int) => x + y			  def addThenMultiply(x: Int, y: Int)(multiplier: Int): Int = (x + y) * multiplier
		  println(add(1, 2))							  println(addThenMultiply(1, 2)(3))


														  def getSquareString(input: Double): String = {
														    val square = input * input
														    square.toString
														  }


函数和方法的区别：

·	函数是一等公民，使用val语句可以定义函数，def语句定义方法；

·	函数是一个对象，继承自FuctionN，函数对象有curried,equals,isInstanceOf,toString等方法，而方法不具有；

·	函数是一个值，可以给val变量赋值，方法不是值、也不可以给val变量赋值；

·	通过将方法转化为函数的方式 method _ 或 method(_) 实现给val变量赋值；

·	若 method 有重载的情况，方法转化为函数时必须指定参数和返回值的类型；

·	某些情况下，编译器可以根据上下文自动将方法转换为函数；

·	无参数的方法 method 没有参数列表（调用：method），无参数的函数 function 有空列表（调用：function()）；

·	方法可以使用参数序列，转换为函数后，必须用 Seq 包装参数序列；

·	方法支持默认参数，函数不支持、会忽略之；	

				
类：																							样本类（case classes)： 使用样本类可以方便的存储和匹配类的内容，不用new关键字就可以创建
使用class关键字定义类，后跟其名称和构造函数参数													默认情况下，样本类是不可变的，并按值进行比较。

class Greeter(prefix: String, suffix: String) {												case class Point(x: Int, y: Int)
  def greet(name: String): Unit = 			// 方法greet的返回类型为Unit，表示返回没有任何意义
    println(prefix + name + suffix)															实例化样本类：
}																							val point = Point(1, 2)
																							val anotherPoint = Point(1, 2)
使用new创建类的实例：																			val yetAnotherPoint = Point(2, 2)
val greeter = new Greeter("Hello, ", "!")																	
greeter.greet("Scala developer") 																	
																							case class Calculator(brand: String, model: String) 
																							val hp20b = Calculator("HP", "20b") 																	
																							样本类主要被设计用在模式匹配中的										
对象：																	
使用object定义对象																	
object IdFactory {																	
  private var counter = 0																	
  def create(): Int = {																	
    counter += 1																	
    counter																	
  }																	
}																	
																	
可以通过引用名称来访问该对象																	
val newId: Int = IdFactory.create()																	
println(newId) // 1																	
val newerId: Int = IdFactory.create()
println(newerId) // 2



抽象类：
不能创建抽象类的实例：
abstract class Shape {      
  def getArea():Int    // subclass should define this      
} 
class Circle(r: Int) extends Shape {      
  def getArea():Int = { r * r * 3 }      
} 
val s = new Shape //error: class Shape is abstract; cannot be instantiated                     
val c = new Circle(2)


特质（Traits）
特质包含某些字段和方法的类型，可以组合多种特质
使用trait定义特质
trait Greeter {
  def greet(name: String): Unit
}

特质的默认实现：
trait Greeter {
  def greet(name: String): Unit =
    println("Hello, " + name + "!")
}

可以使用extends扩展traits并使用override覆盖实现

class DefaultGreeter extends Greeter

class CustomizableGreeter(prefix: String, postfix: String) extends Greeter {
  override def greet(name: String): Unit = {
    println(prefix + name + postfix)
  }
}

val greeter = new DefaultGreeter()
greeter.greet("Scala developer") // Hello, Scala developer!

val customGreeter = new CustomizableGreeter("How are you, ", "?")
customGreeter.greet("Scala developer") // How are you, Scala developer?

·   优先使用特质。一个类扩展多个特质是很方便的，但却只能扩展一个抽象类。
 
·   如果你需要构造函数参数，使用抽象类。因为抽象类可以定义带参数的构造函数，而特质不行。例如，你不能说trait t(i: Int) {}，参数i是非法的。


main方法：程序的切入点
定义方式：
object Main {
  def main(args: Array[String]): Unit =
    println("Hello, Scala developer!")
}


/***********************************************************************************************************************************/
/***********************************************************************************************************************************/

·  类名+括号，调用对象的apply方法 

·  对象名+括号，调用类的apply方法

class ApplyTest {
    println("class ApplyTest")
    def apply() {
        println("class APPLY method")
    }
}
object ApplyTest {
    println("object ApplyTest")
    def apply() = {
        println("object APPLY method")
        new ApplyTest()
    }
}
 
// 对象名+括号，调用类的apply方法
val a1 = new ApplyTest()
a1() // == a1.apply()
// 输出 class ApplyTest, class APPLY method
 
// 类名+括号，调用对象的apply方法
val a2 = ApplyTest()
// 输出 object ApplyTest, object APPLY method, class ApplyTest
 
val a2 = ApplyTest()
a2()
// 输出 object ApplyTest, object APPLY method, class ApplyTest, class APPLY method
 
val a3 = ApplyTest
// 输出 object ApplyTest
 
val a3 = ApplyTest
a3()
// 输出 object ApplyTest, object APPLY method, class ApplyTest



单例对象：用于持有一个类的唯一实例。通常用于工厂模式

object Timer {   
  var count = 0    
  def currentCount(): Long = {     
    count += 1     
    count   
  } 
} 

或者： Timer.currentCount() //1


模式匹配：

匹配值：
val times = 1  
times match {   
  case 1 => "one"   
  case 2 => "two"   
  case _ => "some other number" 
}

使用守卫进行匹配：
times match {   
  case i if i == 1 => "one"   
  case i if i == 2 => "two"   
  case _ => "some other number" 
} 

匹配类型
可以使用match来分别处理不同类型的值
def bigger(o: Any): Any = {   
  o match {     
    case i: Int if i < 0 => i - 1     
    case i: Int => i + 1     
    case d: Double if d < 0.0 => d - 0.1     
    case d: Double => d + 0.1     
    case text: String => text + "s"   
  }
} 

匹配类成员：
定义一个计算器，通过类型对其进行分类
def calcType(calc: Calculator) = calc match {   
  case _ if calc.brand == "HP" && calc.model == "20B" => "financial"   
  case _ if calc.brand == "HP" && calc.model == "48G" => "scientific"   
  case _ if calc.brand == "HP" && calc.model == "30B" => "business"   
  case _ => "unknown" 
} 

样本类基于构造函数的参护士，自动的实现了相等性和易读的toString方法
val hp20b = Calculator("HP", "20b") 
val hp20B = Calculator("HP", "20b") 
hp20b == hp20B // true


样本类主要被设计用在模式匹配中的

val hp20b = Calculator("HP", "20B") 
val hp30b = Calculator("HP", "30B")  
def calcType(calc: Calculator) = calc match {   
  case Calculator("HP", "20B") => "financial"   
  case Calculator("HP", "48G") => "scientific"   
  case Calculator("HP", "30B") => "business"   
  case Calculator(ourBrand, ourModel) => "Calculator: %s %s is of unknown type".format(ourBrand, ourModel)    // 也可以写为： case Calculator(_, _) => "Calculator of unknown type" 
} 																											  // 或者 case _ => "Calculator of unknown type" 
																											  // 或者 case c@Calculator(_, _) => "Calculator: %s of unknown type".format(c) 

映射map， 详细参见： https://blog.csdn.net/cm_chenmin/article/details/52821127

选项Option
Option是一个表示有可能包含值得容器
基本的接口：
trait Option[T] {   
  def isDefined: Boolean   
  def get: T   
  def getOrElse(t: T): T 
} 

本身为泛型，有两个子类： Some[T], None
Map.get 使用 Option 作为其返回值，表示这个方法也许不会返回你请求的值。
val numbers = Map("one" -> 1, "two" -> 2) 
numbers.get("two") // Option[Int] = Some(2)  
numbers.get("three") // Option[Int] = None 


函数组合子（Functional Combinators)
List(1, 2, 3) map squared对列表中的每一个元素都应用了squared平方函数，并返回一个新的列表List(1, 4, 9)。我们把类似于map的操作称作组合子。 他们常被用在标准的数据结构上。

map
map对列表中的每个元素应用一个函数，返回应用后的元素所组成的列表。

val numbers = List(1, 2, 3, 4) 
numbers.map((i: Int) => i * 2) // List[Int] = List(2, 4, 6, 8)

或传入一个函数 (Scala编译器自动把我们的方法转换为函数)
def timesTwo(i: Int): Int = i * 2 
numbers.map(timesTwo) // List[Int] = List(2, 4, 6, 8) 

foreach， 类似于map，但没有返回值。仅作用于有副作用[side-effects]的函数。
numbers.foreach((i: Int) => i * 2)

filter
filter移除任何对传入函数计算结果为false的元素。返回一个布尔值的函数通常被称为谓词函数[或判定函数]
numbers.filter((i: Int) => i % 2 == 0) // List[Int] = List(2, 4) 
def isEven(i: Int): Boolean = i % 2 == 0 
numbers.filter(isEven) // List[Int] = List(2, 4)

List(1, 2, 3).zip(List("a", "b", "c")) // List[(Int, String)] = List((1,a), (2,b), (3,c)) 

val numbers = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) 
numbers.partition(_ % 2 == 0) // (List[Int], List[Int]) = (List(2, 4, 6, 8, 10),List(1, 3, 5, 7, 9)) 

numbers.find((i: Int) => i > 5) // Option[Int] = Some(6) 

numbers.drop(5) // List[Int] = List(6, 7, 8, 9, 10)  会删除前i个元素

dropWhile 将删除匹配谓词函数的第一个元素。例如，如果我们在numbers列表上使用dropWhile函数来去除奇数, 
1将被丢弃（但3不会被丢弃，因为他被2“保护”了）。

numbers.dropWhile(_ % 2 != 0) // List[Int] = List(2, 3, 4, 5, 6, 7, 8, 9, 10) 

foldLeft  以及 foldRight   后者和前者一样，只是运行过程相反
numbers.foldLeft(0)((m: Int, n: Int) => m + n) // Int = 55 

0 为初始值（记住numbers是List[Int]类型），m作为一个累加器。

可视化观察运行过程：

numbers.foldLeft(0) { (m: Int, n: Int) => println("m: " + m + " n: " + n); m + n } 
//m: 0 n: 1 m: 1 n: 2 m: 3 n: 3 m: 6 n: 4 m: 10 n: 5 m: 15 n: 6 m: 21 n: 7 m: 28 n: 8 m: 36 n: 9 m: 45 n: 10 
//res0: Int = 55 

flatten
flatten将嵌套结构扁平化一个层级。
List(List(1, 2), List(3, 4)).flatten // List[Int] = List(1, 2, 3, 4) 


flatMap
flatMap是一种常用的组合子，结合映射[mapping]和扁平化[flattening]。 flatMap需要一个处理嵌套列表的函数，然后将结果串连起来。

val nestedNumbers = List(List(1, 2), List(3, 4)) 
nestedNumbers.flatMap(x => x.map(_ * 2)) // List[Int] = List(2, 4, 6, 8) 

可以把它看做是“先映射后扁平化”的快捷操作：
nestedNumbers.map((x: List[Int]) => x.map(_ * 2)).flatten // List[Int] = List(2, 4, 6, 8)





/***********************************************************************************************************************************/
								模式匹配和函数组合
/***********************************************************************************************************************************/
def f(s: String) = "f(" + s + ")" 
def g(s: String) = "g(" + s + ")" 


1. compose 组合其它函数								andThen 和前者相似，但是调用顺序为先调用第一个函数，后调用第二个函数
val fComposeG = f _ compose g _ 					val fAndThenG = f - andThen g _
fComposeG("yay")    // f(g(yay)) 					fAndThenG("yay")   // g(f(yay))



case 语句是一个名为 PartialFunction（偏函数） 的函数的子类


对于给定的输入参数类型，函数可以接受该类型的任何值；
对于给定的输入参数类型，偏函数只能接受该类型的某些特定的值。 这样使用isDedinedAt来确定
val one: PartialFunction[Int, String] = { case 1 => "one"}
one.isDefinedAt(1)    // true
one.isDefinedAt(2)    // false 

可以直接调用偏函数
one(1)     // one
// 这个应该也可以直接使用 partial(1)