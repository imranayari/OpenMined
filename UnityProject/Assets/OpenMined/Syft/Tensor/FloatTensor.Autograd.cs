using System;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        private bool autograd;
        public FloatTensor Grad { get; private set; }

        private bool keepgrads;

        private List<int> creators;
        private string creation_op;
        private List<int> children_indices; // children -> counts
	    private List<int> children_counts; // children -> counts
	    private int sibling;

        public void InitAutograd()
        {
//			if(!autograd) {
            autograd = true;
            creators = new List<int>();
	        children_indices = new List<int>();
	        children_counts = new List<int>();
	        
//			}
        }

	    public void ResetAutogradCounts()
	    {
		    for (int i = 0; i < children_counts.Count; i++)
		    {
			    children_counts[i] = 0;
		    }
		    
	    }

        public bool AllChildrenGradsAccountedFor()
        {
            foreach (var item in children_counts)
            {
                if (item == 0)
                {
                    return false;
                }
            }
            return true;
        }


        // hook autograd two parents - one scalar
        public void HookAutograd(ref FloatTensor result, float x, string creation_op)
        {
            if (autograd)
            {
                FloatTensor new_child =
                    new FloatTensor(_controller: controller, _shape: new int[] {1}, _data: new float[] {x});

                result.InitAutograd();
                result.creators.Add(this.id);
                result.creators.Add(new_child.id);
                result.creation_op = creation_op;
		        
                children_indices.Add(result.Id);
	            children_counts.Add(0);
//				new_child.children.Add (result.Id, 0);
			}

		}

		// hook autograd two parents
		public void HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op) {

			if (autograd) {

				result.InitAutograd ();
				result.creators.Add (this.id);
				result.creators.Add (x.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);
				
				x.children_indices.Add(result.Id);
				x.children_counts.Add(0);
				
				this.sibling = x.id;
			}

		}

		// hook autograd single parent
		public void HookAutograd(ref FloatTensor result, string creation_op) {

			if (autograd) {

				result.InitAutograd ();
				result.creators.Add (this.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);
			}
		}

	    public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
	    {

		    if (autograd)
		    {
			    if (grad == null)
			    {
				    grad = this.controller.createOnesTensorLike(this);
				    grad.Autograd = false;
			    }

			    if (grad_origin != null)
			    {
				    int child_index = children_indices.IndexOf(grad_origin.Id);
				    if (children_counts[child_index] > 0)
				    {
					    throw new InvalidOperationException("Can't backprop more than once.");
				    }
				    else
				    {
					    children_counts[child_index] += 1;
				    }
			    }

			    if (this.Grad == null)
			    {
				    this.Grad = grad;
			    }
			    else
			    {
				    this.Grad.Add(grad, true);
			    }

			    // grads must not have grads of their own
			    if (this.Grad.autograd == true)
			    {
				    throw new InvalidOperationException("Sorry, grads cannot have grads");
			    }

			    // only continue backpropping if there's something to backprop into
			    // only continue backpropping if all gradients (from children) are accounted for
			    // override waiting for children if "backprop" was called on this variable directly
			    if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllChildrenGradsAccountedFor()))
			    {
                    if (creation_op == "add_elem")
                    {

                        controller.getTensor(creators[0]).Backward(grad, this);
                        controller.getTensor(creators[1]).Backward(grad, this);

                    }
                    else if (creation_op == "mul_elem")
                    {

                        controller.getTensor(creators[0]).Backward(grad.Mul(creators[1]), this);
                        controller.getTensor(creators[1]).Backward(grad.Mul(creators[0]), this);

                    }
                    else if (creation_op == "div_elem")
                    {

                        controller.getTensor(creators[0]).Backward(grad.Div(creators[1]), this);
                        controller.getTensor(creators[1]).Backward(grad.Div(creators[0]), this);

                    }
                    else if (creation_op == "sub_elem")
                    {

                        controller.getTensor(creators[0]).Backward(grad, this);
                        controller.getTensor(creators[1]).Backward(grad.Neg(), this);

                    }
                    else if (creation_op == "mm")
                    {

                        controller.getTensor(creators[0]).Backward(grad.MM(controller.getTensor(creators[1]).Transpose()), this);
                        controller.getTensor(creators[1]).Backward(controller.getTensor(creators[0]).Transpose().MM(grad), this);

                    }
                    else if (creation_op == "sigmoid")
                    {

                        FloatTensor c = this.Copy();
                        c.autograd = false;
                        controller.getTensor(creators[0]).Backward(c.Neg().Add((float)1).Mul(this).Mul(grad), this);

                    }
                    else if (creation_op == "pow_scalar")
                    {

                        FloatTensor self_nograd = controller.getTensor(creators[0]).Copy();
                        self_nograd.autograd = false;
                        controller.getTensor(creators[0]).Backward(self_nograd.Mul(grad).Mul(controller.getTensor(creators[1]).Data[0]), this);

                    }
                    else if (creation_op == "softmax")
                    {
                        FloatTensor c = this.Copy();
                        FloatTensor gradients = new FloatTensor(_controller: controller, _shape: c.shape);

                        for (var i = 0; i < c.shape[0]; i++)
                        {
                            for (var j = 0; j < c.shape[1]; j++)
                            {
                                if (i == j)
                                {
                                    gradients[i, j] = c[i, j] * (1 - c[i, j]);
                                }
                                else
                                {
                                    gradients[i, j] = c[i, j] * c[j, i];
                                }
                            }
                        }
                        controller.getTensor(creators[0]).Backward(gradients.Mul(grad));
                    }
			    }
		    }
	    }
    }
}