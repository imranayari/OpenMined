using System;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

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
        public FloatTensor HookAutograd(ref FloatTensor result, float x, string creation_op, bool inline)
        {

	        if (inline)
		        return this;
	        
	        bool autograd_pre_initialized = false;
            
	        if (result == null)
	        {
		        if (this.children_indices.Count > 0)
		        {
			        autograd_pre_initialized = true;
			        result = controller.getTensor(this.children_indices[0]);
			        result.Zero_();
		        }
		        else
		        {
			        result = this.emptyTensorCopy();
		        }
	        }

	        if (autograd_pre_initialized)
	        {
		        this.ResetAutogradCounts();
		        result.ResetAutogradCounts();
	        }
	        else if (autograd)
	        {
		        FloatTensor new_child =
			        new FloatTensor(_controller: controller, _shape: new int[] {1}, _data: new float[] {x});

		        result.InitAutograd();
		        result.creators.Add(this.id);
		        result.creators.Add(new_child.id);
		        result.creation_op = creation_op;
		        
		        children_indices.Add(result.Id);
		        children_counts.Add(0);
	        }

	        return result;
        }

		// hook autograd two parents
		public FloatTensor HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op, bool inline=false, int[] resultShape= null)
		{

			if (inline)
				return this;
		
			// checks to see if the input has been seen previously. If so, then it assumes
			// that we should just use the previous computation graph instead of initializing
			// a new result. The assumption here is that if the same tensors are used to perform
			// the same operation, then they should output to the same memory instead of allocating
			// new memory.
			bool autograd_pre_initialized = false;

			if (result == null)
			{
				if (this.sibling == x.id)
				{
					//Debug.Log("Id:" + this.id + " Children:" + this.children_indices.Count);
					autograd_pre_initialized = true;
					result = controller.getTensor(this.children_indices[0]);
					result.Zero_();
				}
				else
				{
					if (resultShape != null)
					{
						result = new FloatTensor(_controller: controller, _shape: resultShape);
					}
					else
					{
						result = this.emptyTensorCopy();
					}

					if (this.dataOnGpu)
						result.Gpu(shader);

				}
			}

			if (autograd_pre_initialized)
			{
				this.ResetAutogradCounts();
				result.ResetAutogradCounts();
				x.ResetAutogradCounts();


			}
			else if (autograd)
			{
				result.InitAutograd();
				result.creators.Add(this.id);
				result.creators.Add(x.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);

				x.children_indices.Add(result.Id);
				x.children_counts.Add(0);

				this.sibling = x.id;
			}
		

			return result;

		}

		// hook autograd single parent
		public FloatTensor HookAutograd(ref FloatTensor result, string creation_op, bool inline=false) {

			if (inline)
				return this;
			
			bool autograd_pre_initialized = false;
			//Debug.Log("Id:" + this.id + " Children:" + this.children.Count);
			if (result == null)
			{
				if (autograd && this.children_indices.Count > 0)
				{
					autograd_pre_initialized = true;
					result = controller.getTensor(this.children_indices[0]);
					result.Zero_();
				}
				else
				{
					result = this.emptyTensorCopy();
				}
			}
			
			if (autograd_pre_initialized)
			{
				this.ResetAutogradCounts();
				result.ResetAutogradCounts();
			}
			else if (autograd)
			{
				
				result.InitAutograd ();
				result.creators.Add (this.id);
				result.creation_op = creation_op;

				children_indices.Add(result.Id);
				children_counts.Add(0);
			}

			return result;

		}
    }
}