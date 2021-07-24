console.log('lets begin to code jane');

"use strict";


   
class Synaptic_Matrix
{
  //constructor function
  constructor(rows,cols)
  {
    this.rows=rows;
    this.cols=cols;
    
    this.matrix=[];
  
    for(let i=0;i<this.rows;i++)
      {
        this.matrix[i]=[];
        
        for(let j=0;j<this.cols;j++)
          {
               this.matrix[i][j]=0;
          }
      }
  }
  
  
  //randomize function
  randomize()
  {
    for(let i=0;i<this.rows;i++)
    {
      for(let j=0;j<this.cols;j++)
        {
           this.matrix[i][j]=Math.random(-1,1)*2-1;
        }
    }
  }
  

//scalar addition function || elementwise addition function (depending on the inputs)

  addition(n)
  {
     if(n instanceof Synaptic_Matrix)
       {
         for(let i=0;i<this.rows;i++)
           {
             for(let j=0;j<this.cols;j++)
                {
                   this.matrix[i][j]+=n.matrix[i][j];
                }
           }
        }
      else
        {
          for (let i = 0; i < this.rows; i++)
            {
             for (let j = 0; j < this.cols; j++)
               {
                  this.matrix[i][j] += n;
               }
            }
        }
  }
  
  
  //static function for scalar and element wise addition

 static addition(m,n)
  { 
    let arr= new Synaptic_Matrix(m.rows,m.cols);
    
  
    if (n instanceof Synaptic_Matrix)
    { 
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
           arr.matrix[i][j]= m.matrix[i][j] + n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j]+n;
        }
      }
    }
    return arr;
  }

  
  
  subtraction(n)
  {
     if(n instanceof Synaptic_Matrix)
       {
         for(let i=0;i<this.rows;i++)
           {
             for(let j=0;j<this.cols;j++)
                {
                   this.matrix[i][j]-=n.matrix[i][j];
                }
           }
        }
      else
        {
          for (let i = 0; i < this.rows; i++)
            {
             for (let j = 0; j < this.cols; j++)
               {
                  this.matrix[i][j] -= n;
               }
            }
        }
  }
  
  
  //static function for scalar and element wise subtraction

 static subtraction(m,n)
  { 
    let arr= new Synaptic_Matrix(m.rows,m.cols);
   
  
    if (n instanceof Synaptic_Matrix)
    { 
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
           arr.matrix[i][j]= m.matrix[i][j] - n.matrix[i][j];
        }
      }
    }
    else
    {
      for (let i = 0; i < m.rows; i++)
      {
        for (let j = 0; j < m.cols; j++)
        {
          arr.matrix[i][j] = m.matrix[i][j]-n;
        }
      }
    }
    return arr;
  }

  //matrix multiplication function.
  
  
  multiplication(n)
{
  if(n instanceof Synaptic_Matrix)
  {
    //vector multiplication done
    if(this.cols!==n.rows)
    { 
      console.error("Columns of A must be equal to rows if B");
      return undefined;
    }
    
    let a = this;
    let b = n;
    
    let result= new Synaptic_Matrix(a.rows,b.cols);
    
    for(let i=0;i<result.rows;i++)
    { 
      for(let j=0;j<result.cols;j++)
      { 
        let sum=0;
        for(let k=0;k<a.cols;k++)
        { 
       //   console.log(b.matrix[k][j]);
          
          sum += a.matrix[i][k]*b.matrix[k][j];
        }
   
        result.matrix[i][j]=sum;
      }
    }
   
    return result;
    
  }
  else
  { 
    //scalar multiplication done 
  for (let i = 0; i < this.rows; i++)
  {
    for (let j = 0; j < this.cols; j++)
    {
      this.matrix[i][j] *= n;
    }
  }
  }
}


  // static hadamard elementwise multiplication in an array.
  
  static multiplicationElementwise(m,n)
  { 
    let newArr= new Synaptic_Matrix(m.rows,m.cols);
    
    if(n instanceof Synaptic_Matrix)
  {
    for(let i=0;i<m.rows;i++)
    {
      for(let j=0;j<m.cols;j++)
      {
        newArr.matrix[i][j]=m.matrix[i][j]*n.matrix[i][j]; 
      }
    }
  }
  else
  {
    for (let i = 0; i < m.rows; i++)
    {
      for (let j = 0; j < m.cols; j++)
      {
        newArr.matrix[i][j] = m.matrix[i][j] * n;
      }
    }
  }
  
    return newArr;
  }
  
  
 //transpose function
static transpose(arr)
{ 
  var result= new Synaptic_Matrix(arr.cols,arr.rows);
  for (let i = 0; i < arr.rows; i++)
  {
    for (let j = 0; j < arr.cols; j++)
    {
      result.matrix[j][i]=arr.matrix[i][j];
    }
  }
  return result;
}


//mapping a function
map(func)
{
  for (let i = 0; i < this.rows; i++)
  {
    for (let j = 0; j < this.cols; j++)
    {
      let t= this.matrix[i][j] ;
      this.matrix[i][j]=func(t);
    }
  }
}



//static map function.
static map(arr,func)
{ 
  
  let newArr=new Synaptic_Matrix(arr.rows,arr.cols);
  for (let i = 0; i < arr.rows; i++)
  {
    for (let j = 0; j < arr.cols; j++)
    {
      let t = arr.matrix[i][j];
      newArr.matrix[i][j] = func(t);
    }
  }
  return newArr;
}


//converting an array to a matrix.
static fromArray(arr)
{
  let m= new Synaptic_Matrix(arr.length,1);
  
  for(let i=0;i<arr.length;i++)
  {
    m.matrix[i][0]=arr[i];
  }
  
  return m;
}



//converting  a matrix to  an array
static toArray(mat)
{
  let arr=[];
  
  for(let i=0;i<mat.rows;i++)
  {
    for(let j=0;j<mat.cols;j++)
    {
      arr.push(mat.matrix[i][j]);
    }
  }
  return arr;
}



// static function for printing the array 
 static print(arr)
{
  console.log(arr.matrix);
  console.log(arr.rows);
  console.log(arr.cols);
  
}

 
  
}






class Jane_Activation
{
  static activation_sigmoid(x)
  {
    return (1/(1+Math.exp(-x)));
  }
  
  static derevative_sigmoid(y)
  {
    return y*(1-y);
  }
  
  static activation_relu(x)
  {
    return Math.max(x,0);
  }
  
  static derevative_relu(y)
  {
    if(y===0) return 0;
    return 1;
  }
 
}





class  Jane_FFNN
{
  constructor(inputs,outputs,hidden_layer_array,weights,bias)
  {
    
    hidden_layer_array.push(outputs);
    
    this.output_nodes=outputs;
    this.inputs_nodes=inputs;
    
    
   // Synaptic_Matrix.print(this.input_nodes);
    
    
    this.hidden=hidden_layer_array.length;
    
    let col=this.inputs_nodes;
 
    this.weights=[];
    
    if(weights===undefined)
    {
    for(let i=0;i<hidden_layer_array.length;i++)
    {
      let row=hidden_layer_array[i];
      
      let weighted_array=new Synaptic_Matrix(row,col);
      weighted_array.randomize();
      
      this.weights.push(weighted_array);
      col=row;
    }
    }
    
    else
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let row = hidden_layer_array[i];
      
        let weighted_array = weights[i];
      
        this.weights.push(weighted_array);
        col = row;
      }
      
     // Synaptic_Matrix.print(this.weights[i]);
    } 
      
      
    this.bias=[];
    
    if(bias===undefined)
    {
    for(let i=0;i<hidden_layer_array.length;i++)
      {
        let layer_bias=new Synaptic_Matrix(hidden_layer_array[i],1);
        layer_bias.randomize();
        
        this.bias.push(layer_bias);
        
       // Synaptic_Matrix.print(this.bias[i]);
      }
    }
    else
    {
      for (let i = 0; i < hidden_layer_array.length; i++)
      {
        let layer_bias = bias[i];
      
        this.bias.push(layer_bias);
      
        // Synaptic_Matrix.print(this.bias[i]);
      }
    }
      
    this.learning_rate=0.1;
  }
  
  
  
  weighted_sum(input,weight)
  {
   let output=weight.multiplication(input);
   
   //Synaptic_Matrix.print(output);
   
   return output;
  }
  
  
  transpose(input)
  {
    let output=new Synaptic_Matrix(input.cols,input.rows);
    for(let i=0;i<input.rows;i++)
    {
      for(let j=0;j<input.cols;j++)
      {
        output.matrix[j][i]=input.matrix[i][j];
      }
    }
 //   Synaptic_Matrix.print(output)
    return output;
  }
  
  
  
  
  
  feedforward(inputs_array)
  {
    let layered_weighted_sum=[];
   
 //   let input_temp=Synaptic_Matrix.toArray(inputs_array)
    
    
    let input=Synaptic_Matrix.fromArray(inputs_array);
    
   /// Synaptic_Matrix.print(input);
   
   
    layered_weighted_sum.push(input);
    
    for(let i=1;i<this.hidden+1;i++)
    {
      layered_weighted_sum.push(this.weighted_sum(input,this.weights[i-1]));
      
     // Synaptic_Matrix.print(layered_weighted_sum[i]);
      
      
      layered_weighted_sum[i].addition(this.bias[i-1]);
      
      
      layered_weighted_sum[i]=Synaptic_Matrix.map(layered_weighted_sum[i],Jane_Activation.activation_sigmoid);
      
      
      input=layered_weighted_sum[i];
      
      
  // Synaptic_Matrix.print(layered_weighted_sum[i]);

    }
   // console.log(layered_weighted_sum.length)
    return layered_weighted_sum;
  }
  
  
  
  
  backpropagation(inputs_array,answer_array)
  {
    let right_output=Synaptic_Matrix.fromArray(answer_array);
    
    
    let feedforward=this.feedforward(inputs_array);
    
    let error_in_output=Synaptic_Matrix.subtraction(right_output,feedforward[feedforward.length-1]);
    
    
   // Synaptic_Matrix.print(error_in_output);
    
    
    
    let layered_errors=[];
    let error=error_in_output;
    let error_prev_layer;
    layered_errors.push(error);
    
    

    for(let i=0;i<this.hidden;i++)
    {
      let weight_ith_layer_transposed=this.transpose(this.weights[this.hidden-i-1]);
      
      ///Synaptic_Matrix.print(weight_ith_layer_transposed);
      
       error_prev_layer=weight_ith_layer_transposed.multiplication(error);
      
      layered_errors.push(error_prev_layer);
      
      error=error_prev_layer;
      
    // Synaptic_Matrix.print(error_prev_layer);
     /**/
      
    }
    
   
    for(let i=0;i<feedforward.length;i++)
    {
      let gradient=Synaptic_Matrix.map(feedforward[feedforward.length-1-i],Jane_Activation.derevative_sigmoid);
      
    //  Synaptic_Matrix.print(gradient);
    
      gradient.multiplication(this.learning_rate);
   //   Synaptic_Matrix.print(gradient);
      
     
      gradient=Synaptic_Matrix.multiplicationElementwise(gradient,layered_errors[i]);
      
   //   Synaptic_Matrix.print(gradient)
      /**/
      if(i<feedforward.length-1)
      {
      this.bias[this.bias.length-1-i].addition(gradient);
      }
      
      
   
     let delta_weights;
     if(i<feedforward.length-1)
     { 
      let feedforward_transposed=this.transpose(feedforward[feedforward.length-2-i]);
      
      delta_weights=gradient.multiplication(feedforward_transposed);
      
   //   Synaptic_Matrix.print(delta_weights);
      this.weights[this.weights.length-i-1].addition(delta_weights);
      
      //Synaptic_Matrix.print(this.weights[this.weights.length-i-1]);
     }
   
     
    }
    
    return{
      weights:this.weights,
      errors:layered_errors,
      bias:this.bias
    }
    
    
  }
  
  
  
}





class Jane_CNN
{
  constructor(frames_array, pool_array)
  {

    //frames of the cnn
    this.frames = [];
    for (let i = 0; i < frames_array.length; i++)
    {
      this.frames.push([]);
      for(let j=0;j<frames_array[i].length;j++)
      {
      let a = frames_array[i][j][0];
      let b = frames_array[i][j][1];


      let frame_temp = new Synaptic_Matrix(a, b);
      frame_temp.randomize();


      this.frames[i].push(frame_temp);
      }
    }



    //pooling layers of a cnn
    this.pooling = [];
    for (let i = 0; i < pool_array.length; i++)
    {
      this.pooling.push(pool_array[i]);
     
    }


    this.dense_weights = undefined;
    this.dense_bias = undefined;
    this.learning_rate=.01;

  }






//padding function 


  padding(input,padding_size)
  {
    let output=new Synaptic_Matrix(input.rows+padding_size,input.cols+padding_size);
    
    for(let i=0;i<input.rows;i++)
    {
      for(let j=0;j<input.cols;j++)
      {
        output.matrix[i+1][j+1]=input.matrix[i][j];
      }
    }
    
 //   Synaptic_Matrix.print(output);
   
   return output;
  }
  
  
  
  
  
  //flipping a matrix
  
  
  flip(input)
  {
    let output_temp=new Synaptic_Matrix(input.rows,input.cols)
    for(let i=0;i<input.rows;i++)
    {
      for(let j=0;j<input.cols;j++)
      {
        output_temp.matrix[i][j]=input.matrix[input.rows-i-1][j];
      }
    }
    
    let output=new Synaptic_Matrix(input.rows,input.cols)
    for(let i=0;i<input.rows;i++)
    {
      for(let j=0;j<input.cols;j++)
      {
        output.matrix[i][j]=output_temp.matrix[i][input.rows-j-1];
      }
    }
  //  Synaptic_Matrix.print(input);
  //  Synaptic_Matrix.print(output);
    
    return output;
  }
  
  
  
  
  //submatrix
  
  submatrix(input,i,j,framelength_row,framelength_col)
  {
    let output=new Synaptic_Matrix(framelength_row,framelength_col);
    
    for(let k=i,p=0;p<framelength_row;k++,p++)
    {
      for(let l=j,q=0;q<framelength_col;l++,q++)
      {
        output.matrix[p][q]=input.matrix[k][l];
      }
    }
    
    
   // Synaptic_Matrix.print(output);
   return output;
    
  }
  
  
  
  
  
  
  //weighted_sum 
  
  
  weighted_sum(input,frame)
  {
    
    if(input.rows!==frame.rows||input.cols!==frame.cols)
    {
      console.error("an error in dimensions is occurring");
      
      return undefined;
    }
    
    
    let sum=0;
    
    let hadamard_value=Synaptic_Matrix.multiplicationElementwise(input,frame);
    
    
    for(let i=0;i<hadamard_value.rows;i++)
    {
      for(let j=0;j<hadamard_value.cols;j++)
      {
        sum+=hadamard_value.matrix[i][j];
      }
    }
    
  //  Synaptic_Matrix.print(hadamard_value);
  //  console.log(sum);
    
    return sum;
    
  }
  
  
  
  
  
  
  
  //convolution 
  
  convolution(input,frame)
  { 
    let x=input.rows-frame.rows+1;
    let y=input.cols-frame.cols+1;
    let output=new Synaptic_Matrix(x,y);
    
    for(let i=0;i<(input.rows-frame.rows+1);i++)
    {
      for(let j=0;j<(input.cols-frame.cols+1);j++)
      {
        let sub_matrix=this.submatrix(input,i,j,frame.rows,frame.cols);
        
        let output_temp=this.weighted_sum(sub_matrix,frame);
        
        output.matrix[i][j]=output_temp;
      }
    }
    
   // Synaptic_Matrix.print(output);
    return output;
    
  }
  
  
  
  
  //complete convolution 
  
  complete_convolution(input,frame)
  {
    if(input.cols>1)
    {
   let output = new Synaptic_Matrix(input.rows + 2, input.cols + 2);
    
    for (let i = 1, k = 0; k < input.rows; i++, k++)
    {
      for (let j = 1, l = 0; l < input.cols; j++, l++)
      {
        output.matrix[i][j] = input.matrix[k][l];
      }
    }
    let result = this.convolution(output, frame);
    
   // Synaptic_Matrix.print(result);
    
    return result;
    }
    else
    {
      let output = new Synaptic_Matrix(input.rows + 2, input.cols );
      
      for (let i = 0,k=0; k < input.rows; i++, k++)
      {
        for (let j = 0, l = 0; l < input.cols;l++,j++)
        {
          output.matrix[i][j] = input.matrix[k][l];
        }
      }
      
     // Synaptic_Matrix.print(output);
      
      let result = this.convolution(output, frame);
      
     //  Synaptic_Matrix.print(result);
      
      return result;
    }
  }
  
  
  
  
  
  
  //pooling function
  
  pool(input,pooling)
  {
    let pool_row = Math.floor(input.rows %pooling[0]);
    let pool_col= Math.floor(input.cols % pooling[1]);
    
    
    
    if (pooling[0] === undefined || pooling[0] === 0 || pooling[1]===undefined || pooling[1]===0)
    { 
      return input;
    }
    
    else
    {
    
      let input_temp = new Synaptic_Matrix(input.rows + pool_row, input.cols+pool_col);
  
      for (let i = 0; i < input.rows; i++)
      {
        for (let j = 0; j < input.cols; j++)
        {
          input_temp.matrix[i][j] = input.matrix[i][j];
        }
      }
      input = input_temp;
    
 
   
   let  new_matrix = new Synaptic_Matrix(input
   .rows-pooling[0]+1, input.cols-pooling[1]+1);
  
  //Synaptic_Matrix.print(new_matrix);
    
    for (let i = 0, p = 0; i < input.rows; i+=pooling[0], p++)
    {
      for (let j = 0, q = 0; j < input.cols; j += pooling[1], q++)
      {
        let maximum = -1000000;
        for (let k = 0; k < pooling[0]; k++)
        {
          for (let l = 0; l < pooling[1]; l++)
          {
            if (input.matrix[k + i][l + j] > maximum)
            { 
            //  console.log('**')
              maximum = input.matrix[k + i][l + j];
            }
          }
        }
      //
     //  console.log(p,q)
        new_matrix.matrix[p][q] = maximum;
        
      }
    }
    
   //Synaptic_Matrix.print(new_matrix);
    return new_matrix;
    
   }
  }
  
  
  
  
  
  depool(input,pooling)
  {
    if(pooling[0]===0 && pooling[1]===0)
    {
      return input;
    }
    
    
   
    let  output=new Synaptic_Matrix(input.rows+pooling[0]-1,input.cols+pooling[1]-1)
    

     
    
    
    for(let i=0,p=0;i<output.rows;i+=pooling[0],p++)
    {
      for(let j=0,q=0;j<output.cols;j+=pooling[1],q++)
      {
        output.matrix[i][j]=input.matrix[p][q];
      }
    }
    
   // Synaptic_Matrix.print(output);
    
    
    return output;
  }
  
  
  feedforward(inputs_array)
  {
    let inputs=inputs_array;
    let convolution_layers=[];
    
    for(let k=0;k<this.frames.length;k++)
    { 
       convolution_layers.push([]);
       for(let i=0;i<inputs.length;i++)
        {
           for(let j=0;j<this.frames[k].length;j++)
              {
                let conv_temp=this.convolution(inputs[i],this.frames[k][j]);
                
                conv_temp=Synaptic_Matrix.map(conv_temp,Jane_Activation.activation_relu);
                
                conv_temp=this.pool(conv_temp,this.pooling[k]);
                
                convolution_layers[k].push(conv_temp);
                
              //  Synaptic_Matrix.print(conv_temp);
              }
        }
        inputs=convolution_layers[k];
       // console.error(inputs.length)
    }
    return convolution_layers;
  }
  
  
  
  cnn_to_ffnn(input,correct_answers,hidden_layer)
  {
    
    
    let finput_temp=input[input.length-1];
    let finput=[];
    
    for(let i=0;i<finput_temp.length;i++)
    {
      let temp_val=Synaptic_Matrix.toArray(finput_temp[i]);
   
      for(let j=0;j<temp_val.length;j++)
      {
        finput.push(temp_val[j]);
      }
      
    }
    
   // console.error(finput.length);
   // console.log(finput);

     let dense_layer=new Jane_FFNN(finput.length,correct_answers.length,hidden_layer,this.dense_weights,this.dense_bias);
     
     let dense_layer_value=dense_layer.backpropagation(finput,correct_answers);
     
     this.dense_bias=dense_layer_value.bias;
     this.dense_weights=dense_layer_value.weights;
     
     let dense_error_obtained=dense_layer_value.errors;
     
     let error_cnn_op=dense_error_obtained[dense_error_obtained.length-1]
     
  //   Synaptic_Matrix.print(error_cnn_op);
     
     
 
     let last_cnn_layer_error=[];
     let range=input.length-1
     let count=0;
     
     for(let i=0;i<input[range].length;i++)
     {
       let temp_matrix=new Synaptic_Matrix(input[range][i].rows,input[range][i].cols);
       
       for(let j=0;j<temp_matrix.rows;j++)
       {
         for(let k=0;k<temp_matrix.cols;k++)
         {
           temp_matrix.matrix[j][k]=error_cnn_op.matrix[count][0];
           count++;
         }
       }
       last_cnn_layer_error.push(temp_matrix);
       
     }
     
     
 
  //  console.log(last_cnn_layer_error.length)
 
     return last_cnn_layer_error;
     
     
  }
  
  average_error(front_section,back_section,framelength)
  { 
  
      let desired_rows=back_section.rows;
      let desired_cols=back_section.cols;
    
    //console.error(desired_rows,desired_cols)
    
       let avg_err;
     
      for(let j=0;j<framelength;j++)
      { 
         
       
          avg_err = new Synaptic_Matrix(desired_rows,desired_cols);
          
      
        
        let temp_arr=new Synaptic_Matrix(desired_rows,desired_cols);
        
        
        for(let k=0;k<desired_rows;k++)
        {
          for(let l=0;l<desired_cols;l++)
          { 
           if(k<front_section[j].rows && l<front_section[j].cols)
           {
            temp_arr.matrix[k][l]=front_section[j].matrix[k][l];
            
           }
            
          }
          
        }
       // console.error(j);
         
       
          avg_err=Synaptic_Matrix.addition(avg_err,temp_arr);
         /// Synaptic_Matrix.print(avg_err);
      } 
     /* */
    
    
   // Synaptic_Matrix.print(avg_err);
       
    return avg_err;
  }


  backpropagation(input,feedforward_input,initial_input)
  {
    let layered_errors=[];
    layered_errors.push(input)
    
    for(let i=this.frames.length-1;i>0;i--)
    {
      let j=0;
      let jc=0;
      let depooled_errors=[];
      
      while(j<input.length)
      { 
        let temp_error_stack=[];
        
        for(let k=0;k<this.frames[i].length;k++)
        {
          let temp_complete_conv=this.complete_convolution(input[j],this.flip(this.frames[i][k]));
          
          
      //  Synaptic_Matrix.print(temp_complete_conv);
        
          
          temp_error_stack.push(temp_complete_conv);
          
          j++;
        }
        
      // console.log(feedforward_input[i-1].length,jc)
       
   //   Synaptic_Matrix.print(feedforward_input[i-1][jc]);
      
       
       let pooled_errors=this.average_error(temp_error_stack,feedforward_input[i-1][jc],this.frames[i].length);
       
       
      // Synaptic_Matrix.print(pooled_errors);
      // console.table(this.pooling[i])
       
      
        // let depooled_temp=this.depool(pooled_errors,this.pooling[i-1]);
         
     //    Synaptic_Matrix.print(depooled_temp);
         
         depooled_errors.push(pooled_errors);
       
       
       jc++;
      
      /**/
      }
      
      input=depooled_errors;
      layered_errors.push(input);
      
      
    }
    
    
    let layer_gradient_cnn=[];
   // console.error(feedforward_input.length-1)
    
    for(let i=feedforward_input.length-1,p=0;i>=0;i--,p++)
    {
      let input_gr;
      input_gr=feedforward_input[i];
      layer_gradient_cnn.push([]);
      
      for(let j=0;j<input_gr.length;j++)
      {
        let feedforward_derevative_map=Synaptic_Matrix.map(input_gr[j],Jane_Activation.derevative_relu);
        
      
    //  Synaptic_Matrix.print(initial_input[0]);
        
       //console.error(initial_input[i].length);
        
     //  Synaptic_Matrix.print(layered_errors[layered_errors.length-1-i][j])
        
     
       
       let gradient=Synaptic_Matrix.multiplicationElementwise(feedforward_derevative_map,layered_errors[layered_errors.length-1-i][j]);
        
      // Synaptic_Matrix.print(gradient);
       
       layer_gradient_cnn[p].push(gradient);
      
      }
   //   console.error(layer_gradient_cnn[p].length)
      
    }
    
    
    let delta_frames=[];
    
    for(let i=feedforward_input.length-1;i>=0;i--)
    { 
      delta_frames.push([]);
      
      let inp;
      if(i>0)
      {
        inp=feedforward_input[i-1];
      }
      else
      {
        inp=initial_input;
      }
      let temp_frame;
      let f=feedforward_input.length-1;
      
      
      for(let j=0;j<this.frames[i].length;j++)
      { 
        temp_frame=[];
         
        for(let k=0;k<inp.length;k++)
        { 
          
      
         
        // console.error(inp.length); 
        //  Synaptic_Matrix.print(inp[k])
       //  console.error(k*this.frames[i+1].length)
       //  console.warn(this.frames[i+1].length)
       //  console.warn(layer_gradient_cnn[f-i].length,i)
       //  Synaptic_Matrix.print(layer_gradient_cnn[layer_gradient_cnn.length-i-1][j+k*this.frames[i+1].length])
          
       //   console.log(j,k)
       
          let gradient_error=this.convolution(inp[k],layer_gradient_cnn[f-i][j+k*this.frames[i].length]);
          
         //Synaptic_Matrix.print(gradient_error);
      
       temp_frame.push(gradient_error);
        }
        
        
        //Synaptic_Matrix.print(this.frames[i+1][j]);
    ///    console.log(temp_frame.length);
     //   console.error(inp.length)
      
        let temp_frame_value=this.average_error(temp_frame,this.frames[i][j],inp.length);
        
      // Synaptic_Matrix.print(this.frames[i][j]);
     //   console.error("***");
        this.frames[i][j].addition(temp_frame_value);
        
        Synaptic_Matrix.print(this.frames[i][j]);
      }
    }/**/
    
    
    console.log("training completed");
    return this.frames;
    
    
  }



}