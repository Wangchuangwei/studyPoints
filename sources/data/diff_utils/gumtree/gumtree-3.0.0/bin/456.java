public class Test {
public int hashCode() { 
    int result = name.hashCode(); 
    result = 32 * result + 2 * zScore.hashCode(); ; 
    return result; 
}
 }